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

    # --- Dizinler (modul seviyesinden erisilebilir kisayollar) ---
    BASE_DIR = BASE_DIR
    DATA_DIR = DATA_DIR
    MODELS_DIR = MODELS_DIR
    OUTPUT_DIR = OUTPUT_DIR

    # --- Model Yollari ---
    # Trafik modulu icin nesne tespit modeli (araclar).
    # yolo11s (small): nano'dan belirgin daha dogru sinif tahmini
    # (otomobil/kamyon/otobus karismasi azalir), CPU'da hala kabul edilebilir.
    TRAFFIC_MODEL_PATH = MODELS_DIR / "yolo11s.pt"
    # Plaka tespiti icin ozel YOLO modeli (license_plate sinifi).
    PLATE_MODEL_PATH = MODELS_DIR / "license_plate_detector.pt"
    # Surucu modulu yuz landmark modeli (MediaPipe Tasks "FaceLandmarker").
    # 478 landmark (iris dahil) -> goz/agiz/kafa sinyalleri buradan hesaplanir.
    # .task dosyasi Google'in modelleri olup bir kez indirilir (driver_module
    # ilk calistiginda yoksa otomatik indirir).
    DRIVER_MODEL_PATH = MODELS_DIR / "face_landmarker.task"
    DRIVER_MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    )

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

    # --- Performans Modlari ---
    # CPU'da hiz/dogruluk dengesi. UI'dan ya da --perf ile secilir.
    #   fast     : en hizli (kucuk model + kucuk imgsz)
    #   balanced : varsayilan denge
    #   accurate : en dogru (buyuk imgsz)
    # Plaka tespiti dengeli/dogru modda HER karede yapilir: akici videolarda
    # hizli gecen araclari kacirmamak icin (OCR butcesi zaten "kaleci" onceligi
    # ile sinirlanir). Yalnizca fast modda 2 karede bir tespit edilir.
    PERF_MODE = "balanced"
    PERF_PRESETS = {
        "fast":     {"track_model": "yolo11n.pt", "track_imgsz": 480,
                     "plate_detect_interval": 2},
        "balanced": {"track_model": "yolo11s.pt", "track_imgsz": 512,
                     "plate_detect_interval": 1},
        "accurate": {"track_model": "yolo11s.pt", "track_imgsz": 640,
                     "plate_detect_interval": 1},
    }

    # COCO veri setinde arac sayilan sinif id'leri:
    # 2: car, 3: motorcycle, 5: bus, 7: truck
    VEHICLE_CLASSES = [2, 3, 5, 7]

    # --- Plaka Okuma (OCR / ANPR) Ayarlari ---
    # Plaka tespit modelinin minimum guven skoru. Dusuk tutmak, uzak/kucuk
    # plakalari da yakalar (daha cok arac okunur); gurultu okumalar oylama +
    # bicim dogrulamasi ile elenir.
    PLATE_CONFIDENCE_THRESHOLD = 0.20
    # Plaka tespiti HER karede degil, bu kadar karede bir yapilir (CPU tasarrufu;
    # perf moduna gore PERF_PRESETS tarafindan ezilir). OCR zaten kisitli oldugu
    # icin her kare tespit gereksizdir.
    PLATE_DETECT_INTERVAL = 2
    # EasyOCR dil listesi ('en' Latin harf/rakamlari kapsar; TR plakalar da Latin).
    OCR_LANGUAGES = ["en"]
    # GPU yoksa CPU'da calisir (bu makinede CUDA yok -> False).
    OCR_USE_GPU = False
    # Plaka kirpintisi OCR'dan once bu HEDEF GENISLIGE buyutulur/kuculur.
    # ~200px EasyOCR icin hem yeterince keskin hem de hizlidir.
    OCR_TARGET_WIDTH = 200
    # Ayni arac icin en az bu kadar kare gecmeden OCR tekrar denenmez (CPU dostu).
    # Not: "kareden cikmaya yakin" (acil) araclar bu araligi atlayip her karede
    # okunur -> cikmadan once mumkun oldugunca cok oy toplanir.
    OCR_REATTEMPT_INTERVAL = 6
    # Tek bir karede en fazla kac plaka OCR'a sokulsun. "Kaleci" onceligiyle
    # bunlar daima kareden CIKMAYA EN YAKIN araclardir (2-3 arac yan yana ise
    # hepsi okunur).
    OCR_MAX_PER_FRAME = 3
    # Bir aracin "acil" (kareden cikmak uzere) sayilmasi icin tahmini cikis
    # suresi esigi (kare cinsinden). Bu surenin altindaki araclar her karede,
    # araligi beklemeden okunur.
    OCR_URGENT_FRAMES = 18
    # Gecerli sayilacak min plaka karakter sayisi (gurultu metni eler).
    # TR/AB plakalari her zaman >=5 karakter -> 4 ve altini ele (orn. "K5ZK").
    OCR_MIN_PLATE_CHARS = 5
    # Tek bir OCR parcasinin oya katilmasi icin gereken min guven.
    OCR_FRAGMENT_MIN_CONF = 0.10
    # Plaka tespit kutusu cogu zaman karakterlere YAPISIK gelir; ozellikle
    # hareketli araclarda kutu biraz kayinca soldaki haneler ("34" gibi) kirpilir.
    # OCR'dan once kutu, kendi GENISLIK/YUKSEKLIGININ bu oraninda her yone genisletilir
    # (kareyi tasmadan kirpilir). Yatayda biraz daha fazla pay, kenar karakterleri kurtarir.
    OCR_CROP_PAD_X = 0.12
    OCR_CROP_PAD_Y = 0.22

    # --- Plaka Dogrulama (Validation) Ayarlari ---
    # Plaka EKRANDA gosterilmeden once en az bu kadar okuma (oy) birikmeli.
    # 1 = ilk okumada hemen goster (dusuk gecikme; bicim duzeltmesi ilk okumayi
    # da temizler). Plaka "kilitlenince" donderilir ve bir daha degismez.
    OCR_MIN_VOTES_TO_SHOW = 1
    # Plaka metni bilinen bir ulke BICIMINE uyuyorsa oyu bu kat ile carpilir.
    # Boylece bicimsel olarak gecerli okumalar gecersizleri yener.
    OCR_VALID_FORMAT_BONUS = 2.0

    # --- Plaka Oylama / Kilit (Kararlilik) Ayarlari ---
    # Plaka metni kare kare degil, arac basina agirlikli OYLAMA ile belirlenir.
    # Yeterli ve tutarli oy birikince plaka KILITLENIR: o andaki uzlasi metni
    # donderilir, OCR durur (butce diger araclara kalir) ve metin DEGISMEZ.
    # Kilit icin gereken toplam (agirlikli) oy skoru.
    OCR_LOCK_SCORE = 1.6
    # Lider metnin ikinciden bu kat fazla oyu varsa kilit kabul edilir.
    OCR_LOCK_RATIO = 1.6

    # --- Plaka Gorunumu (Hologram) ---
    # Hologram panelinde plakanin solundaki mavi banda yazilacak ulke kodu.
    # Turk plakalari icin "TR"; bu UK test videosu icin "GB" yapabilirsin.
    PLATE_COUNTRY_CODE = "TR"
    # Ulke kodu nasil belirlensin:
    #   "auto"  -> plaka bicimine bakarak tahmin et (karisik trafik icin),
    #              karar verilemezse PLATE_COUNTRY_CODE'a duser.
    #   "force" -> her plakaya PLATE_COUNTRY_CODE'u yaz (tek-ulkeli videolarda
    #              en guvenilir; UI'dan TR/GB secilince bu kullanilir).
    PLATE_COUNTRY_MODE = "auto"

    # --- Hiz Olcumu Ayarlari ---
    # Goruntudeki piksel mesafesini gercek dunyaya cevirmek icin kalibrasyon
    # (metre / piksel). Sahaya gore kalibre edilmelidir.
    PIXELS_PER_METER = 8.0
    SPEED_LIMIT_KMH = 50

    # --- Sürücü Modülü (Uyku/Dikkat) Ayarları ---
    # Sürücü içi kamera kaynağı (webcam indeksi). run_driver.py --cam ile ezilebilir.
    DRIVER_CAM_SOURCE = 0

    # Göz Açıklık Oranı (EAR - Eye Aspect Ratio): göz dikey/yatay mesafe oranı.
    # Göz kapandıkça düşer. Bu eşiğin ALTINDA ise göz "kapalı" sayılır.
    EYE_AR_THRESHOLD = 0.25
    # Göz bu kadar ARDIŞIK kare kapalı kalırsa "microsleep" -> ALARM tetiklenir
    # (~30 FPS'te 20 kare ≈ 0.7 sn göz kapalı).
    DROWSINESS_FRAMES = 20

    # PERCLOS: son penceredeki gözün kapalı kaldığı kare oranı (yorgunluğun
    # en güvenilir göstergesi). Pencere uzunluğu (saniye) ve "yorgun" eşik oranı.
    PERCLOS_WINDOW_SEC = 30.0
    PERCLOS_DROWSY_RATIO = 0.40   # Karelerin %40'ında göz kapalıysa DROWSY uyarısı

    # Ağız Açıklık Oranı (MAR - Mouth Aspect Ratio): esneme tespiti.
    # Bu eşiğin ÜSTÜNDE ve yeterince uzun sürerse bir esneme sayılır.
    MAR_YAWN_THRESHOLD = 0.6
    YAWN_MIN_FRAMES = 12          # Esnemenin geçerli sayılması için min ardışık kare

    # Kafa eğikliği (pitch, derece): baş öne düşerse (uyuklama) artan değer.
    # Bu açının üstünde ve yeterince uzun sürerse "baş düşmesi" (nod) sayılır.
    HEAD_PITCH_THRESHOLD = 18.0
    NOD_MIN_FRAMES = 12

    # ALARM durumunda sesli uyarı (Windows'ta yerleşik winsound.Beep ile;
    # winsound yoksa sessizce atlanır). run_driver.py --no-sound ile kapatılır.
    DRIVER_ALERT_SOUND = True
    DRIVER_ALERT_BEEP_HZ = 1100   # Beep frekansı (Hz)
    DRIVER_ALERT_BEEP_MS = 600    # Beep süresi (ms)

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
