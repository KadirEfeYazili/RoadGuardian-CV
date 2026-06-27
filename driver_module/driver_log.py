"""
RoadGuardian-CV - Sürücü Olay Kayıt (Event Log) Modülü

Sürücü durumundaki önemli olayları (ALARM, YORGUN, esneme, baş düşmesi) kalıcı
bir dosyaya (CSV ya da JSON Lines) yazar. Böylece sürücü modülü yalnızca canlı
gösterim değil, sonradan incelenebilen bir KAYIT da üretir (ne zaman, hangi
karede, hangi olay, EAR/PERCLOS ne idi).

Tasarım plate_log.py ile aynıdır:
- Dosya biçimi uzantıdan seçilir: ``.jsonl`` -> JSON Lines, aksi halde CSV.
- CSV, TR yereli Excel ile düzgün açılsın diye NOKTALI VİRGÜL (';') ayıraç ve
  UTF-8 BOM ile yazılır; Türkçe karakterler bozulmaz.
- Her satır anında diske yazılır (flush); çalışma yarıda kesilse de kayıt kalır.
- Aynı olay her karede tekrar yazılmaz: yalnızca durumun YÜKSELEN KENARI (olayın
  başladığı an) kaydedilir (run_driver.py edge tespitiyle çağırır).
"""

import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.config import settings  # noqa: E402

# CSV ayıracı: TR yereli Excel ';' bekler (',' kullanınca her şey tek sütuna düşer).
_CSV_DELIM = ";"

_FIELDS = [
    "timestamp",      # gerçek dünya zamanı (ISO) - canlı kamera için anlamlı
    "frame",          # video/kare indeksi
    "video_time",     # andaki süre (mm:ss) - kare/FPS ile hesaplanır
    "event",          # olay türü: ALARM / DROWSY / YAWN / NOD
    "reason",         # insan-okur kısa gerekçe
    "ear",            # o andaki göz açıklık oranı
    "perclos",        # o andaki PERCLOS (%)
    "closed_frames",  # ardışık kapalı kare sayısı
]


@dataclass
class DriverEvent:
    """Kaydedilecek tek bir sürücü olayı."""

    frame: int
    event: str
    reason: str
    ear: float
    perclos: float
    closed_frames: int

    def as_row(self, fps: float) -> dict:
        secs = self.frame / fps if fps else 0.0
        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "frame": self.frame,
            "video_time": f"{int(secs // 60):02d}:{int(secs % 60):02d}",
            "event": self.event,
            "reason": self.reason,
            "ear": round(float(self.ear), 3),
            "perclos": round(float(self.perclos) * 100, 1),
            "closed_frames": int(self.closed_frames),
        }


class DriverLogger:
    """Sürücü olaylarını CSV/JSONL dosyasına yazar.

    Args:
        path: Çıktı dosyası. Uzantı ``.jsonl`` ise JSON Lines, değilse CSV.
        fps: Video kare hızı (video_time hesabı için); verilmezse config FPS.
    """

    def __init__(self, path, fps: float | None = None):
        self.path = Path(path)
        self.fps = float(fps if fps is not None else settings.FPS)
        self.is_jsonl = self.path.suffix.lower() == ".jsonl"
        self._count = 0

        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists() or self.path.stat().st_size == 0
        if self.is_jsonl:
            encoding = "utf-8"
        else:
            encoding = "utf-8-sig" if is_new else "utf-8"
        self._fh = open(self.path, "a", newline="", encoding=encoding)
        if not self.is_jsonl and is_new:
            csv.DictWriter(
                self._fh, fieldnames=_FIELDS, delimiter=_CSV_DELIM
            ).writeheader()
            self._fh.flush()

    def log(self, event: DriverEvent) -> bool:
        """Olayı dosyaya yazar (her zaman yazar; tekrar engelleme çağırana ait)."""
        row = event.as_row(self.fps)
        if self.is_jsonl:
            self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            # TR Excel ondalık ayıracı virgül bekler: sayılar metin değil sayı görünsün.
            row = {
                **row,
                "ear": str(row["ear"]).replace(".", ","),
                "perclos": str(row["perclos"]).replace(".", ","),
            }
            csv.DictWriter(
                self._fh, fieldnames=_FIELDS, delimiter=_CSV_DELIM
            ).writerow(row)
        self._fh.flush()
        self._count += 1
        return True

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    @property
    def count(self) -> int:
        """Yazılan toplam olay sayısı."""
        return self._count
