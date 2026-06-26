"""
RoadGuardian-CV - Sürücü Yüz Landmark Okuyucu (MediaPipe Face Mesh)

MediaPipe Tasks "FaceLandmarker" modelini sarmalar. Her kareden 478 yüz
landmark'ı (iris dahil) çıkarır ve bunlardan sürücü uyku/dikkat sinyallerini
hesaplar:

    - EAR (Eye Aspect Ratio)   -> gözün ne kadar açık olduğu (kapanma tespiti)
    - MAR (Mouth Aspect Ratio) -> ağız açıklığı (esneme tespiti)
    - pitch (derece)           -> başın öne eğikliği (uyuklama / baş düşmesi)

Model dosyası (.task) yoksa ilk çalıştırmada Google'ın sunucusundan bir kez
otomatik indirilir; sonraki çalıştırmalar yereldeki dosyayı kullanır.

Not: Bu sürümde (mediapipe 0.10.x) eski "solutions.face_mesh" API'si kaldırıldı;
güncel ve desteklenen "Tasks" API'si kullanılır.
"""

import math
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.config import settings  # noqa: E402

# --- Yüz Mesh landmark indeksleri (MediaPipe Face Mesh, 478 nokta) ---
# Her göz için EAR'da kullanılan 6 nokta: [köşe1, üst1, üst2, köşe2, alt2, alt1].
# EAR = (|üst1-alt1| + |üst2-alt2|) / (2 * |köşe1-köşe2|)
_RIGHT_EYE = (33, 160, 158, 133, 153, 144)   # görüntüde sol taraftaki göz
_LEFT_EYE = (362, 385, 387, 263, 373, 380)   # görüntüde sağ taraftaki göz

# Ağız (MAR): üst-iç dudak, alt-iç dudak ve iki ağız köşesi.
_MOUTH_TOP = 13
_MOUTH_BOTTOM = 14
_MOUTH_LEFT = 61
_MOUTH_RIGHT = 291


@dataclass
class FaceSignals:
    """Bir kareden çıkarılan sürücü sinyalleri."""
    found: bool          # Karede yüz bulundu mu?
    ear: float           # Göz açıklık oranı (iki gözün ortalaması)
    mar: float           # Ağız açıklık oranı
    pitch: float         # Başın öne eğikliği (derece; öne eğildikçe artar)


def _to_pixels(landmarks, indices, w, h):
    """Normalize landmark'ları piksel koordinatına çevirir (Nx2 dizi)."""
    return np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in indices],
        dtype=np.float32,
    )


def _eye_aspect_ratio(pts):
    """Tek bir göz için EAR. pts: [köşe1, üst1, üst2, köşe2, alt2, alt1]."""
    vertical = np.linalg.norm(pts[1] - pts[5]) + np.linalg.norm(pts[2] - pts[4])
    horizontal = np.linalg.norm(pts[0] - pts[3])
    if horizontal < 1e-6:
        return 0.0
    return float(vertical / (2.0 * horizontal))


def eye_aspect_ratio(landmarks, w, h):
    """İki gözün ortalama EAR değeri."""
    right = _eye_aspect_ratio(_to_pixels(landmarks, _RIGHT_EYE, w, h))
    left = _eye_aspect_ratio(_to_pixels(landmarks, _LEFT_EYE, w, h))
    return (right + left) / 2.0


def mouth_aspect_ratio(landmarks, w, h):
    """Ağız açıklık oranı (dikey açıklık / yatay genişlik). Esnemede yükselir."""
    pts = _to_pixels(
        landmarks, (_MOUTH_TOP, _MOUTH_BOTTOM, _MOUTH_LEFT, _MOUTH_RIGHT), w, h
    )
    vertical = np.linalg.norm(pts[0] - pts[1])
    horizontal = np.linalg.norm(pts[2] - pts[3])
    if horizontal < 1e-6:
        return 0.0
    return float(vertical / horizontal)


def head_pitch(matrix):
    """
    Yüz dönüşüm matrisinden (4x4) baş eğikliğini (pitch, derece) çıkarır.

    matrix: MediaPipe'ın verdiği yüz->kamera dönüşüm matrisi (numpy 4x4) ya da
    None. Dönüş: başın öne/arkaya eğikliği (derece). Yoksa 0.0.
    """
    if matrix is None:
        return 0.0
    r = np.asarray(matrix, dtype=np.float32)[:3, :3]
    # Standart dönme matrisi -> Euler ayrıştırması (x ekseni etrafı = pitch).
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    return float(pitch)


class FaceMeshReader:
    """
    MediaPipe FaceLandmarker sarmalayıcısı (VIDEO modu).

    Kullanım:
        reader = FaceMeshReader()
        signals = reader.process(frame_bgr, timestamp_ms)
        if signals.found: ...
        reader.close()
    """

    def __init__(self, model_path=None):
        # İçe aktarmayı sınıf içinde yap: mediapipe ağır bir bağımlılık ve
        # ilk import yüklemesi yavaş; modül import edilince beklememek için.
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        model_path = Path(model_path) if model_path else settings.DRIVER_MODEL_PATH
        _ensure_model(model_path)

        # Modeli yoldan değil, BAYT TAMPONUNDAN yükle: proje yolu Türkçe karakter
        # içerebilir ve MediaPipe'ın C++ dosya açıcısı ASCII olmayan yolları açamaz.
        # Python dosyayı sorunsuz okur; baytları model_asset_buffer ile veririz.
        model_bytes = model_path.read_bytes()

        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=model_bytes),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            # Kafa eğikliği için yüz dönüşüm matrisini de iste.
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def process(self, frame_bgr, timestamp_ms):
        """Bir BGR kareyi işler ve FaceSignals döndürür."""
        h, w = frame_bgr.shape[:2]
        # MediaPipe RGB bekler.
        rgb = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame_bgr[:, :, ::-1]),
        )
        result = self._landmarker.detect_for_video(rgb, int(timestamp_ms))
        if not result.face_landmarks:
            return FaceSignals(found=False, ear=0.0, mar=0.0, pitch=0.0)

        landmarks = result.face_landmarks[0]
        matrix = None
        if result.facial_transformation_matrixes:
            matrix = result.facial_transformation_matrixes[0]
        return FaceSignals(
            found=True,
            ear=eye_aspect_ratio(landmarks, w, h),
            mar=mouth_aspect_ratio(landmarks, w, h),
            pitch=head_pitch(matrix),
        )

    def close(self):
        self._landmarker.close()


def _ensure_model(model_path):
    """Model .task dosyası yoksa config'deki URL'den bir kez indirir."""
    model_path = Path(model_path)
    if model_path.exists():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Yüz landmark modeli bulunamadi, indiriliyor:\n  {settings.DRIVER_MODEL_URL}")
    urllib.request.urlretrieve(settings.DRIVER_MODEL_URL, str(model_path))
    print(f"Model kaydedildi: {model_path}  ({model_path.stat().st_size} bytes)")
