"""
RoadGuardian-AI - Merkezi Ayar Dosyasi

Projenin tum modulleri (traffic_module, driver_module, api, ui) bu dosyadaki
ayarlari kullanir. Yollar, proje kok dizinine gore otomatik hesaplanir; bu
sayede proje baska bir bilgisayara tasinsa bile calismaya devam eder.
"""

from pathlib import Path

# --- Proje Dizinleri ---
# Bu dosya: <PROJE_KOK>/core/config.py  ->  iki ust dizin proje koku.
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"


class Config:
    """Projenin genel ayarlarini tutan merkezi sinif."""

    # --- Model Yollari ---
    # Trafik modulu icin nesne tespit modeli (araclar).
    TRAFFIC_MODEL_PATH = MODELS_DIR / "yolo11n.pt"
    # Plaka tespiti icin opsiyonel ozel model (ileride egitilecek).
    PLATE_MODEL_PATH = MODELS_DIR / "plate_detector.pt"
    # Surucu modulu icin yuz/goz tespit modeli.
    DRIVER_MODEL_PATH = MODELS_DIR / "driver_face.pt"

    # --- Video / Kaynak Yollari ---
    TRAFFIC_VIDEO_PATH = DATA_DIR / "test_traffic.mp4"
    DRIVER_VIDEO_PATH = DATA_DIR / "test_driver.mp4"

    # Kamera kaynaklari (0 = varsayilan webcam, RTSP/USB icin degistirilebilir).
    EXTERNAL_CAMERA_SOURCE = 0   # Dis kamera (trafik)
    INTERNAL_CAMERA_SOURCE = 1   # Ic kamera (surucu)

    # --- Kamera / Goruntu Ayarlari ---
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    FPS = 30

    # --- Tespit / Takip Ayarlari ---
    CONFIDENCE_THRESHOLD = 0.5   # Minimum tespit guven skoru
    IOU_THRESHOLD = 0.5          # NMS icin IoU esigi
    TRACKER_CONFIG = "bytetrack.yaml"  # YOLO yerlesik tracker konfigurasyonu

    # COCO veri setinde arac sayilan sinif id'leri:
    # 2: car, 3: motorcycle, 5: bus, 7: truck
    VEHICLE_CLASSES = [2, 3, 5, 7]

    # --- Hiz Olcumu Ayarlari ---
    # Goruntudeki piksel mesafesini gercek dunyaya cevirmek icin kalibrasyon
    # (metre / piksel). Sahaya gore kalibre edilmelidir.
    PIXELS_PER_METER = 8.0
    SPEED_LIMIT_KMH = 50

    # --- Surucu Modulu (Uyku/Dikkat) Ayarlari ---
    EYE_AR_THRESHOLD = 0.25      # Goz kapanma orani esigi (EAR)
    DROWSINESS_FRAMES = 20       # Bu kadar kare boyunca goz kapali ise uyari

    # --- API Ayarlari ---
    API_HOST = "0.0.0.0"
    API_PORT = 8000

    # --- Genel ---
    DEBUG = True


# Modullerin "from core.config import settings" seklinde kullanabilmesi icin
# hazir bir ornek.
settings = Config()


def ensure_directories() -> None:
    """Cikti/model/veri dizinleri yoksa olusturur."""
    for directory in (DATA_DIR, MODELS_DIR, OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
