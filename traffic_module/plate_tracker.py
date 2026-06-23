"""
RoadGuardian-CV - Plaka Takip / Onbellek Modulu

Arac takibi (TrafficTracker) ile plaka okuma (PlateReader) arasindaki kopru.

Iki temel fikir uzerine kuruludur:

1) "KALECI" ONCELIGI (goalkeeper scheduling)
   Araclar bir akista kareye girip cikar; bir plaka yalnizca arac kareden
   CIKANA KADAR okunabilir. Bu yuzden sinirli OCR butcesi her zaman kareden
   CIKMAYA EN YAKIN araclara ayrilir (bir kalecinin daima kaleye en yakin topa
   bakmasi gibi). Cikis yakinligi, aracin HIZINDAN tahmini "cikisa kalan kare"
   ile olculur. Yan yana birden cok arac cikmak uzereyse hepsi okunur.

2) AGIRLIKLI OYLAMA + KILIT (voting & lock)
   Her okuma, OCR guveni VE plaka bicimi gecerliligi ile agirliklanir; her
   karakter konumu icin en cok agirlik alan karakter secilir. Plaka okunur
   okunmaz gosterilir (dusuk gecikme); yeterli/tutarli oy birikince KILITLENIR:
   metin bicime gore son kez duzeltilir, DONDURULUR (bir daha degismez) ve o
   arac icin OCR durur (butce diger araclara kalir).
"""

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.config import settings  # noqa: E402
from traffic_module.plate_ocr import (  # noqa: E402
    PlateReader,
    correct_by_format,
    plate_format_score,
)

# Arac basina saklanan max okuma sayisi (bellek sinirlama).
_MAX_READS = 40
# Bir aracin "hareketli" sayilmasi icin min hiz (px/kare); altinda titreme.
_MIN_SPEED = 0.5


@dataclass
class PlateRecord:
    """Bir arac ID'sine ait plaka okuma gecmisi, oylama ve KILIT durumu.

    Gosterilen plaka tek bir okuma degil, agirlikli oylamanin uzlasisidir:
      1) Her okuma ``guven * (1 + BONUS * bicim_skoru)`` ile agirliklanir;
         gecerli bicimli okumalar (orn. "34KLE88") gurultuyu yener.
      2) Okumalar uzunluga gore gruplanir; toplam agirligi en yuksek uzunluk
         secilir.
      3) O uzunlukta her karakter konumu icin en cok agirlik alan karakter
         secilir ve sonuc plaka bicimine gore son kez duzeltilir.

    Plaka ``locked`` olunca ``frozen`` doldurulur ve ``text`` artik sabit kalir.
    """

    # (metin, ham_guven, agirlik) -> agirlik = guven * bicim bonusu
    reads: list[tuple[str, float, float]] = field(default_factory=list)
    plate_box: tuple[int, int, int, int] | None = None   # son plaka konumu
    last_attempt_frame: int = -10_000
    frozen: str | None = None                            # kilit sonrasi sabit metin

    def add_vote(self, text: str, conf: float) -> None:
        weight = conf * (1.0 + settings.OCR_VALID_FORMAT_BONUS * plate_format_score(text))
        self.reads.append((text, conf, weight))
        if len(self.reads) > _MAX_READS:
            self.reads.pop(0)

    def _length_scores(self) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        for t, _c, w in self.reads:
            scores[len(t)] += w
        return scores

    def _dominant_length(self) -> int | None:
        scores = self._length_scores()
        return max(scores, key=scores.get) if scores else None

    def _consensus(self) -> str:
        """Agirlikli karakter oylamasiyla ham uzlasi metni (duzeltme oncesi)."""
        dom = self._dominant_length()
        if not dom:
            return ""
        chars = []
        for i in range(dom):
            col: dict[str, float] = defaultdict(float)
            for t, _c, w in self.reads:
                if len(t) == dom:
                    col[t[i]] += w
            chars.append(max(col, key=col.get))
        return "".join(chars)

    @property
    def text(self) -> str:
        """Gosterilecek plaka: kilitliyse sabit, degilse bicime gore duzeltilmis."""
        if self.frozen is not None:
            return self.frozen
        return correct_by_format(self._consensus())

    @property
    def conf(self) -> float:
        """Uzlasi uzunlugundaki okumalarin gorulen en yuksek HAM guveni."""
        dom = self._dominant_length()
        if not dom:
            return 0.0
        group = [c for t, c, _w in self.reads if len(t) == dom]
        return max(group) if group else 0.0

    @property
    def votes(self) -> int:
        """Uzlasi uzunlugunu destekleyen okuma (oy) sayisi."""
        dom = self._dominant_length()
        if not dom:
            return 0
        return sum(1 for t, _c, _w in self.reads if len(t) == dom)

    @property
    def ready(self) -> bool:
        """Plaka ekranda gosterilmeye hazir mi? (en az N oy + makul uzunluk)."""
        if self.frozen is not None:
            return True
        return (
            self.votes >= settings.OCR_MIN_VOTES_TO_SHOW
            and len(self._consensus()) >= settings.OCR_MIN_PLATE_CHARS
        )

    @property
    def locked(self) -> bool:
        """Kilitli mi? (kalici metin donduruldu)."""
        return self.frozen is not None

    def _lock_ready(self) -> bool:
        """Kilit sarti: baskin uzunluk yeterince ve rakibinden belirgin ustun."""
        if len(self.reads) < 3:
            return False
        scores = sorted(self._length_scores().values(), reverse=True)
        top = scores[0]
        second = scores[1] if len(scores) > 1 else 0.0
        return top >= settings.OCR_LOCK_SCORE and top >= settings.OCR_LOCK_RATIO * second

    def commit(self) -> None:
        """Kilit sarti saglandiysa metni dondur (bir daha degismez)."""
        if self.frozen is None and self._lock_ready():
            text = correct_by_format(self._consensus())
            if len(text) >= settings.OCR_MIN_PLATE_CHARS:
                self.frozen = text


@dataclass
class PlateTracker:
    """Arac ID -> plaka eslemesini, oylamayi ve "kaleci" OCR onceligini yonetir."""

    reader: PlateReader = field(default_factory=PlateReader)
    reattempt_interval: int = settings.OCR_REATTEMPT_INTERVAL
    max_per_frame: int = settings.OCR_MAX_PER_FRAME
    detect_interval: int = settings.PLATE_DETECT_INTERVAL
    urgent_frames: float = settings.OCR_URGENT_FRAMES

    def __post_init__(self):
        self.records: dict[int, PlateRecord] = {}
        self._centers: dict[int, tuple[float, float]] = {}   # son merkez
        self._vel: dict[int, tuple[float, float]] = {}       # EMA hiz (px/kare)
        self._frame_idx = 0

    # --- geometri yardimcilari ------------------------------------------ #
    @staticmethod
    def _center(box) -> tuple[float, float]:
        x1, y1, x2, y2 = box[:4]
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @staticmethod
    def _contains(vbox, point) -> bool:
        x1, y1, x2, y2 = vbox
        px, py = point
        return x1 <= px <= x2 and y1 <= py <= y2

    @staticmethod
    def _area(box) -> float:
        x1, y1, x2, y2 = box[:4]
        return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))

    def _match_plate_to_vehicle(self, plate_box, vehicles):
        """Bir plaka kutusunu iceren en KUCUK (en spesifik) araca esler."""
        center = self._center(plate_box)
        best_id, best_area = None, float("inf")
        for tid, vbox in vehicles.items():
            if self._contains(vbox, center):
                area = self._area(vbox)
                if area < best_area:
                    best_area, best_id = area, tid
        return best_id

    # --- "kaleci" onceligi: cikisa kalan sure --------------------------- #
    def _update_motion(self, vehicles: dict[int, tuple]) -> None:
        """Her arac icin merkez + EMA hizini (px/kare) gunceller."""
        for tid, box in vehicles.items():
            cx, cy = self._center(box)
            prev = self._centers.get(tid)
            if prev is not None:
                inst = (cx - prev[0], cy - prev[1])
                old = self._vel.get(tid, inst)
                self._vel[tid] = (
                    0.6 * old[0] + 0.4 * inst[0],
                    0.6 * old[1] + 0.4 * inst[1],
                )
            self._centers[tid] = (cx, cy)

    def _frames_to_exit(self, tid: int, box, w: int, h: int) -> float:
        """Aracin HAREKET YONUNDE kareden cikmasina kalan tahmini kare sayisi.

        Hareketin onde gelen kenari (gidis yonundeki kenar) ile o yondeki kare
        sinirinin arasindaki mesafe / hiz. Durağan araclar icin sonsuz (acil degil).
        """
        vx, vy = self._vel.get(tid, (0.0, 0.0))
        x1, y1, x2, y2 = box
        tte = float("inf")
        if vx > _MIN_SPEED:
            tte = min(tte, (w - x2) / vx)
        elif vx < -_MIN_SPEED:
            tte = min(tte, x1 / (-vx))
        if vy > _MIN_SPEED:
            tte = min(tte, (h - y2) / vy)
        elif vy < -_MIN_SPEED:
            tte = min(tte, y1 / (-vy))
        return max(0.0, tte)

    def _due_for_ocr(self, rec: PlateRecord) -> bool:
        """Yeniden-deneme araligi doldu mu? (acil araclar bunu atlar)."""
        return self._frame_idx - rec.last_attempt_frame >= self.reattempt_interval

    # ------------------------------------------------------------------ #
    def update(self, frame, vehicles: dict[int, tuple]) -> dict[int, PlateRecord]:
        """Bir kareyi isler: hareketi gunceller, plakalari okur (kaleci onceligi).

        Args:
            frame: Islenecek (temiz) BGR kare.
            vehicles: {track_id: (x1, y1, x2, y2)} -> o karedeki araclar.

        Returns:
            Gosterime hazir {track_id: PlateRecord} kayitlari.
        """
        self._frame_idx += 1
        h, w = frame.shape[:2]

        # Hareket/hiz HER karede guncellenir (cikis tahmini icin), tespit atlasa bile.
        self._update_motion(vehicles)

        # CPU tasarrufu (yalnizca fast modda): plaka tespiti N karede bir.
        if self.detect_interval > 1 and (self._frame_idx % self.detect_interval) != 0:
            return {tid: r for tid, r in self.records.items() if r.ready}

        # 1) Plakalari tespit et ve her birini bir araca esle (ID basina en guvenli).
        matched: dict[int, tuple[int, int, int, int]] = {}
        plate_conf: dict[int, float] = {}
        for (x1, y1, x2, y2, conf) in self.reader.detect_plates(frame):
            tid = self._match_plate_to_vehicle((x1, y1, x2, y2), vehicles)
            if tid is None:
                continue
            if conf >= plate_conf.get(tid, 0.0):
                matched[tid] = (x1, y1, x2, y2)
                plate_conf[tid] = conf

        # 2) Aday sec: kilitli OLMAYAN ve (araligi dolmus VEYA cikmak uzere) araclar.
        candidates = []  # (tte, votes, -plate_conf, tid, pbox)
        for tid, pbox in matched.items():
            rec = self.records.get(tid)
            if rec is None:
                rec = PlateRecord()
                self.records[tid] = rec
            rec.plate_box = pbox  # huzme cizimi icin konumu daima guncelle
            if rec.locked:
                continue  # kilitli: metni sabit, OCR butcesini bosa harcama
            tte = self._frames_to_exit(tid, vehicles[tid], w, h)
            urgent = tte < self.urgent_frames
            if urgent or self._due_for_ocr(rec):
                candidates.append((tte, rec.votes, -plate_conf.get(tid, 0.0), tid, pbox))

        # KALECI ONCELIGI: kareden CIKMAYA EN YAKIN (en kucuk tte) once; esitlikte
        # az okunmus (dusuk oy) ve yuksek plaka guvenli olan one gecer.
        candidates.sort(key=lambda c: c[:3])

        # 3) Butce kadarini oku, oy ekle, kilit sartini kontrol et.
        for _tte, _v, _nc, tid, pbox in candidates[: self.max_per_frame]:
            rec = self.records[tid]
            rec.last_attempt_frame = self._frame_idx
            reading = self.reader.read_plate_from_box(frame, pbox)
            if reading is None:
                continue
            rec.add_vote(*reading)
            rec.commit()  # yeterince kararliysa metni dondur

        # 4) Yalnizca gosterime hazir kayitlari dondur.
        return {tid: r for tid, r in self.records.items() if r.ready}

    def get(self, track_id: int) -> PlateRecord | None:
        rec = self.records.get(track_id)
        return rec if (rec and rec.ready) else None

    def cleanup(self, active_ids: set[int]) -> None:
        """Ekrandan cikan araclarin kayit/hareket verilerini temizler (bellek)."""
        for tid in [t for t in self.records if t not in active_ids]:
            del self.records[tid]
        for tid in [t for t in self._centers if t not in active_ids]:
            del self._centers[tid]
            self._vel.pop(tid, None)
