"""
RoadGuardian-AI - Plaka Okuma (ANPR / OCR) Modulu

Iki asamali bir plaka okuma hatti saglar:

    1. Plaka TESPITI  -> ozel egitilmis YOLO modeli (models/license_plate_detector.pt)
                         tum kare uzerinde plaka kutularini bulur.
    2. Plaka OKUMA    -> EasyOCR, her plakayi buyutup on isleyerek
                         icindeki metni cikartir.

Tasarim notlari:
- CPU'da OCR pahalidir. Bu yuzden okuma "hat" (pipeline) icinde dogrudan degil,
  ``PlateTracker`` (plate_tracker.py) tarafindan arac ID basina onbelleklenerek
  ve kisitlanarak cagrilir.
- ``read_plate`` tek bir plaka kirpintisi icin (metin, guven) dondurur.
- ``detect_plates`` tum kare uzerinde plaka kutularini dondurur.
"""

import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.config import settings  # noqa: E402

# OCR sonucundan plaka disi karakterleri (bosluk, tire, nokta vb.) ayikla.
_NON_ALNUM = re.compile(r"[^A-Z0-9]")


class PlateReader:
    """Plaka tespiti (YOLO) + metin okuma (EasyOCR) yapan sinif."""

    def __init__(
        self,
        model_path: str | None = None,
        languages: list[str] | None = None,
        gpu: bool | None = None,
        plate_conf: float | None = None,
    ):
        self.plate_conf = (
            plate_conf if plate_conf is not None else settings.PLATE_CONFIDENCE_THRESHOLD
        )
        self.upscale = settings.OCR_UPSCALE
        self.min_chars = settings.OCR_MIN_PLATE_CHARS

        # --- Plaka tespit modeli ---
        from ultralytics import YOLO

        self.model = YOLO(str(model_path or settings.PLATE_MODEL_PATH))

        # --- OCR motoru (EasyOCR) ---
        # Ilk kullanimda tanima modellerini indirir; sonra yerelden yuklenir.
        import easyocr

        self.reader = easyocr.Reader(
            languages or settings.OCR_LANGUAGES,
            gpu=settings.OCR_USE_GPU if gpu is None else gpu,
        )

    # ------------------------------------------------------------------ #
    # 1) Plaka tespiti                                                   #
    # ------------------------------------------------------------------ #
    def detect_plates(self, frame) -> list[tuple[int, int, int, int, float]]:
        """Tum kare uzerinde plakalari tespit eder.

        Returns:
            (x1, y1, x2, y2, conf) demetlerinden olusan liste.
        """
        result = self.model(frame, conf=self.plate_conf, verbose=False)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        out: list[tuple[int, int, int, int, float]] = []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), conf in zip(xyxy, confs):
            out.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
        return out

    # ------------------------------------------------------------------ #
    # 2) Plaka okuma (OCR)                                               #
    # ------------------------------------------------------------------ #
    def _preprocess(self, plate_crop):
        """OCR oncesi kirpintiyi temizler: buyut, gri, kontrast.

        Kucuk/dusuk cozunurluklu plakalarda okuma orani belirgin artar.
        """
        if plate_crop is None or plate_crop.size == 0:
            return None

        h, w = plate_crop.shape[:2]
        if h < 4 or w < 4:
            return None

        # En-boy oranini koruyarak buyut.
        crop = cv2.resize(
            plate_crop,
            (w * self.upscale, h * self.upscale),
            interpolation=cv2.INTER_CUBIC,
        )
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Hafif gurultu azalt + kontrast esitle (CLAHE).
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        return gray

    @staticmethod
    def _clean_text(text: str) -> str:
        """OCR metnini plaka formatina yaklastir: buyuk harf + alfasayisal."""
        return _NON_ALNUM.sub("", text.upper())

    def read_plate(self, plate_crop) -> tuple[str, float] | None:
        """Tek bir plaka kirpintisindan (metin, guven) dondurur.

        Plaka tek satir olsa da EasyOCR onu birden cok kutuya bolebilir
        (orn. "34 ABC" + "123"). Bu yuzden tum parcalar SOLDAN SAGA siralanip
        birlestirilir; boylece plakanin tamami yakalanir.

        Plaka benzeri metin bulunamazsa None doner.
        """
        prepped = self._preprocess(plate_crop)
        if prepped is None:
            return None

        # allowlist: yalnizca plakada gecebilecek karakterler -> daha temiz okuma.
        detections = self.reader.readtext(
            prepped,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            detail=1,
            paragraph=False,
        )
        if not detections:
            return None

        # Parcalari sol-x'e gore sirala, gurultu olanlari (cok dusuk guven) ele.
        frags = []
        for bbox, raw_text, conf in detections:
            cleaned = self._clean_text(raw_text)
            if not cleaned or conf < settings.OCR_FRAGMENT_MIN_CONF:
                continue
            x_left = min(p[0] for p in bbox)
            frags.append((x_left, cleaned, float(conf)))

        if not frags:
            return None
        frags.sort(key=lambda f: f[0])

        text = "".join(f[1] for f in frags)
        # Guven = karakter sayisina gore agirlikli ortalama.
        total_chars = sum(len(f[1]) for f in frags)
        conf = sum(f[2] * len(f[1]) for f in frags) / max(1, total_chars)

        if len(text) < self.min_chars:
            return None
        return text, conf

    def read_plate_from_box(
        self, frame, box: tuple[int, int, int, int]
    ) -> tuple[str, float] | None:
        """Kare + plaka kutusu verilince ilgili kirpintiyi okur."""
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        crop = frame[y1:y2, x1:x2]
        return self.read_plate(crop)
