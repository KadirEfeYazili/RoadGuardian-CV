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
    # Plaka tespiti icin ozel YOLO modeli (license_plate sinifi).
    PLATE_MODEL_PATH = MODELS_DIR / "license_plate_detector.pt"
    # Surucu modulu icin yuz/goz tespit modeli.
    DRIVER_MODEL_PATH = MODELS_DIR / "driver_face.pt"

    # --- Video / Kaynak Yollari ---
    TRAFFIC_VIDEO_PATH = DATA_DIR / "test_traffic.mp4"
    # Plakalarin okunabildigi (araclar kameraya yakin) ANPR test videosu.
    PLATE_VIDEO_PATH = DATA_DIR / "plate_test.mp4"
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

    # --- Plaka Okuma (OCR / ANPR) Ayarlari ---
    # Plaka tespit modelinin minimum guven skoru.
    PLATE_CONFIDENCE_THRESHOLD = 0.3
    # EasyOCR dil listesi ('en' Latin harf/rakamlari kapsar; TR plakalar da Latin).
    OCR_LANGUAGES = ["en"]
    # GPU yoksa CPU'da calisir (bu makinede CUDA yok -> False).
    OCR_USE_GPU = False
    # Plaka kirpintisi OCR'dan once bu kat buyutulur (kucuk plakalar icin sart).
    OCR_UPSCALE = 4
    # OCR sadece bir arac icin guvenilir okuma yapilana kadar denenir; ayrica
    # ayni arac icin en az bu kadar kare gecmeden tekrar denenmez (CPU dostu).
    OCR_REATTEMPT_INTERVAL = 8
    # Tek bir karede en fazla kac plaka OCR'a sokulsun (CPU yukunu sinirlar).
    OCR_MAX_PER_FRAME = 4
    # Bir okumanin "kalici kabul" edilmesi icin gereken min OCR guveni.
    OCR_ACCEPT_CONFIDENCE = 0.45
    # Gecerli sayilacak min plaka karakter sayisi (gurultu metni eler).
    # TR/AB plakalari her zaman >=5 karakter -> 4 ve altini ele (orn. "K5ZK").
    OCR_MIN_PLATE_CHARS = 5
    # Tek bir OCR parcasinin oya katilmasi icin gereken min guven.
    OCR_FRAGMENT_MIN_CONF = 0.10

    # --- Plaka Oylama (Kararlilik) Ayarlari ---
    # Plaka metni kare kare degil, arac basina OYLAMA ile belirlenir: her okuma
    # guveniyle agirlikli oy ekler, en cok oy alan metin gosterilir. Bu sayede
    # gosterim titremez ve dogru plaka zamanla one cikar.
    # Bir plakanin "kilitli" (kararli) sayilmasi icin gereken toplam oy skoru.
    OCR_LOCK_SCORE = 1.6
    # Lider metnin ikinciden bu kat fazla oyu varsa kilit kabul edilir.
    OCR_LOCK_RATIO = 1.6
    # Kilitli plakalar OCR butcesini bosa harcamasin diye daha seyrek yenilenir
    # (yeniden-deneme araligi bu kat ile carpilir).
    OCR_LOCKED_REFRESH_MULT = 6

    # --- Plaka Gorunumu (Hologram) ---
    # Hologram panelinde plakanin solundaki mavi banda yazilacak ulke kodu.
    # Turk plakalari icin "TR"; bu UK test videosu icin "GB" yapabilirsin.
    PLATE_COUNTRY_CODE = "TR"

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
