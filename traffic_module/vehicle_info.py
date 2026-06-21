"""
RoadGuardian-AI - Arac Bilgisi (Tip + Renk) Modulu

Arac TIPI zaten takip modelinden (YOLO/COCO) gelir: car, motorcycle, bus, truck.
Bu modul bu tipleri Turkce etikete cevirir ve aracin BASKIN RENGINI tahmin eder.

Renk tahmini: arac kutusunun ust-orta govde bolgesinden (cam/teker/golge
disinda kalan kaput/tavan) ortanca BGR alinir, HSV'ye cevrilip temel renk
sinifina eslenir. Hafif ve onbelleklenebilir; ozel model gerektirmez.
"""

import cv2
import numpy as np

# COCO arac sinif adlari -> Turkce etiket.
TYPE_NAMES_TR = {
    "car": "OTOMOBIL",
    "motorcycle": "MOTOSIKLET",
    "bus": "OTOBUS",
    "truck": "KAMYON",
}

# Temel renk paleti: ad -> temsili BGR (kart uzerindeki ornek kare icin).
_COLOR_BGR = {
    "Siyah": (30, 30, 30),
    "Beyaz": (245, 245, 245),
    "Gri": (128, 128, 128),
    "Gumus": (190, 190, 190),
    "Kirmizi": (40, 40, 210),
    "Turuncu": (30, 130, 240),
    "Sari": (40, 220, 230),
    "Yesil": (60, 170, 70),
    "Mavi": (200, 110, 40),
    "Lacivert": (110, 60, 20),
}


def type_label(class_name: str) -> str:
    """COCO sinif adini Turkce arac tipine cevirir."""
    return TYPE_NAMES_TR.get(class_name, class_name.upper())


def estimate_color(vehicle_crop) -> tuple[str, tuple[int, int, int]]:
    """Arac kirpintisindan baskin rengi tahmin eder.

    Returns:
        (renk_adi, ornek_bgr)
    """
    if vehicle_crop is None or vehicle_crop.size == 0:
        return "Bilinmiyor", (128, 128, 128)

    h, w = vehicle_crop.shape[:2]
    if h < 6 or w < 6:
        return "Bilinmiyor", (128, 128, 128)

    # Ust-orta govde bolgesi (cam/teker/zemin gurultusunu azaltir).
    y1, y2 = int(h * 0.15), int(h * 0.55)
    x1, x2 = int(w * 0.25), int(w * 0.75)
    region = vehicle_crop[y1:y2, x1:x2]
    if region.size == 0:
        region = vehicle_crop

    # Gray-world beyaz dengesi: sahnedeki mavi/soguk renk yansimasini giderir
    # (yoksa gri/gumus araclar yanlislikla "mavi/lacivert" okunur).
    reg = region.astype(np.float32)
    means = reg.reshape(-1, 3).mean(axis=0)  # B, G, R ortalamalari
    gray_mean = float(means.mean())
    scale = gray_mean / (means + 1e-6)
    reg = np.clip(reg * scale, 0, 255).astype(np.uint8)

    # Ortanca BGR (aykiri parlama/golgeye dayanikli).
    b, g, r = (int(np.median(reg[:, :, i])) for i in range(3))
    hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
    hue, sat, val = int(hsv[0]), int(hsv[1]), int(hsv[2])

    # Once akromatik (renksiz) durumlar: parlakliga gore siyah/gri/beyaz.
    # Trafik cogunlukla akromatiktir; sat esigi yuksek tutulur.
    if val < 60:
        name = "Siyah"
    elif sat < 65:
        if val > 200:
            name = "Beyaz"
        elif val > 145:
            name = "Gumus"
        else:
            name = "Gri"
    else:
        # Renkli: OpenCV hue 0..179.
        if hue < 10 or hue >= 170:
            name = "Kirmizi"
        elif hue < 20:
            name = "Turuncu"
        elif hue < 35:
            name = "Sari"
        elif hue < 85:
            name = "Yesil"
        elif hue < 100:
            name = "Mavi"
        elif hue < 130:
            name = "Lacivert" if val < 150 else "Mavi"
        else:
            name = "Kirmizi"

    return name, _COLOR_BGR.get(name, (b, g, r))
