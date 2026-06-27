"""
RoadGuardian-CV - Uyku/Yorgunluk Durum Makinesi

FaceMeshReader'dan gelen kare kare sinyalleri (EAR / MAR / pitch) alır ve
sürücünün durumunu üç kademede özetler:

    AWAKE  (uyanık)  -> her şey normal
    DROWSY (yorgun)  -> uyarı: PERCLOS yüksek, esneme ya da hafif baş düşmesi
    ALARM  (tehlike) -> microsleep (gözler uzun süre kapalı) veya baş tamamen
                        öne düştü -> sesli + görsel ALARM

Ölçütler:
    - Göz kapalı ardışık kare sayısı  -> microsleep (ana alarm)
    - PERCLOS: penceredeki kapalı kare oranı -> yorgunluğun en güvenilir ölçütü
    - Esneme (MAR) ve baş düşmesi (pitch) -> ek yorgunluk işaretleri
"""

import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.config import settings  # noqa: E402

# Durum etiketleri (UI ve kayıt bu sabitleri kullanır).
STATE_NO_FACE = "NO_FACE"
STATE_AWAKE = "AWAKE"
STATE_DROWSY = "DROWSY"
STATE_ALARM = "ALARM"


@dataclass
class DrowsinessState:
    """Bir kare sonrası sürücü durumunun tam özeti."""
    state: str            # STATE_* sabitlerinden biri
    ear: float            # O karedeki göz açıklık oranı
    mar: float            # O karedeki ağız açıklık oranı
    pitch: float          # O karedeki baş eğikliği (derece)
    eye_closed: bool      # Göz şu an kapalı mı?
    closed_frames: int    # Gözün kaç karedir ardışık kapalı olduğu
    perclos: float        # Penceredeki kapalı kare oranı (0..1)
    yawns: int            # Toplam esneme sayısı
    nods: int             # Toplam baş düşmesi sayısı
    alarm: bool           # ALARM (sesli uyarı) tetiklendi mi?
    reason: str           # ALARM/DROWSY gerekçesi (insan-okur kısa metin)


class DrowsinessDetector:
    """
    Kare kare sinyalleri durum makinesine işler.

    Kullanım:
        det = DrowsinessDetector(fps=30)
        state = det.update(ear, mar, pitch, face_found=True)
    """

    def __init__(self, fps=None):
        fps = float(fps or settings.FPS)
        # PERCLOS penceresi: son N saniyeyi kapsayan kayan kare penceresi.
        window = max(1, int(settings.PERCLOS_WINDOW_SEC * fps))
        self._closed_window = deque(maxlen=window)
        # Isınma: pencere bu kadar örnekle dolmadan PERCLOS'a güvenme; aksi halde
        # ilk saniyelerde birkaç blink oranı şişirip yanlış DROWSY verir (~5 sn).
        self._perclos_warmup = max(1, int(fps * 5))

        # Eşikler (config'den; ölü ayar bırakılmaz, hepsi kullanılır).
        self._ear_thr = settings.EYE_AR_THRESHOLD
        self._drowsy_frames = settings.DROWSINESS_FRAMES
        self._perclos_ratio = settings.PERCLOS_DROWSY_RATIO
        self._mar_thr = settings.MAR_YAWN_THRESHOLD
        self._yawn_min = settings.YAWN_MIN_FRAMES
        self._pitch_thr = settings.HEAD_PITCH_THRESHOLD
        self._nod_min = settings.NOD_MIN_FRAMES

        # Sayaçlar / durum.
        self.closed_frames = 0
        self.yawns = 0
        self.nods = 0
        self._yawn_frames = 0
        self._yawn_counted = False
        self._nod_frames = 0
        self._nod_counted = False

    def update(self, ear, mar, pitch, face_found):
        """Bir karelik sinyalleri işler ve güncel DrowsinessState döndürür."""
        if not face_found:
            # Yüz yoksa karar veremeyiz; sayaçları sıfırlamadan NO_FACE bildir.
            # (Geçici kayıpların alarmı tetiklememesi için kapalı-kare sıfırlanır.)
            self.closed_frames = 0
            self._closed_window.append(False)
            return self._make_state(STATE_NO_FACE, 0.0, 0.0, 0.0, False,
                                    False, "Yuz bulunamadi")

        eye_closed = ear < self._ear_thr
        self.closed_frames = self.closed_frames + 1 if eye_closed else 0
        self._closed_window.append(eye_closed)

        # --- Esneme (bir ağız-açma epizodunda bir kez sayılır) ---
        if mar > self._mar_thr:
            self._yawn_frames += 1
            if self._yawn_frames >= self._yawn_min and not self._yawn_counted:
                self.yawns += 1
                self._yawn_counted = True
        else:
            self._yawn_frames = 0
            self._yawn_counted = False

        # --- Baş düşmesi / uyuklama (bir epizodda bir kez sayılır) ---
        nodding = pitch > self._pitch_thr
        if nodding:
            self._nod_frames += 1
            if self._nod_frames >= self._nod_min and not self._nod_counted:
                self.nods += 1
                self._nod_counted = True
        else:
            self._nod_frames = 0
            self._nod_counted = False

        perclos = (
            sum(self._closed_window) / len(self._closed_window)
            if self._closed_window else 0.0
        )

        # --- Durum kararı (öncelik: ALARM > DROWSY > AWAKE) ---
        microsleep = self.closed_frames >= self._drowsy_frames
        head_down = self._nod_frames >= self._nod_min
        yawning = self._yawn_frames >= self._yawn_min

        if microsleep:
            return self._make_state(STATE_ALARM, ear, mar, pitch, eye_closed,
                                    True, "Gozler kapali (microsleep)", perclos)
        if head_down:
            return self._make_state(STATE_ALARM, ear, mar, pitch, eye_closed,
                                    True, "Bas one dustu", perclos)
        perclos_ready = len(self._closed_window) >= self._perclos_warmup
        if perclos_ready and perclos >= self._perclos_ratio:
            return self._make_state(STATE_DROWSY, ear, mar, pitch, eye_closed,
                                    False, "Yuksek PERCLOS (yorgunluk)", perclos)
        if yawning:
            return self._make_state(STATE_DROWSY, ear, mar, pitch, eye_closed,
                                    False, "Esneme", perclos)
        return self._make_state(STATE_AWAKE, ear, mar, pitch, eye_closed,
                                False, "", perclos)

    def _make_state(self, state, ear, mar, pitch, eye_closed, alarm, reason,
                    perclos=0.0):
        return DrowsinessState(
            state=state, ear=ear, mar=mar, pitch=pitch, eye_closed=eye_closed,
            closed_frames=self.closed_frames, perclos=perclos,
            yawns=self.yawns, nods=self.nods, alarm=alarm, reason=reason,
        )
