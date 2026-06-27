"""
RoadGuardian-CV - Sürücü Durumu HUD Overlay

Sürücü uyku/dikkat durumunu, trafik modülünün hologram tarzıyla uyumlu, yarı
saydam bir HUD panelinde gösterir:

    - Sol üstte durum paneli: renk kodlu (yeşil/sarı/kırmızı) durum etiketi
    - EAR ve PERCLOS göstergeleri (çubuk)
    - Esneme ve baş düşmesi sayaçları
    - ALARM durumunda kare kenarında yanıp sönen kırmızı çerçeve + gerekçe

Çizim helper'ları (`_round_rect`, `_corner_brackets`) traffic_module'deki
hologram ile paylaşılır; kod tekrarı yapılmaz.
"""

import sys
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))

from traffic_module.hologram import _round_rect, _corner_brackets, FONT_UI  # noqa: E402
from driver_module.drowsiness import (  # noqa: E402
    STATE_NO_FACE, STATE_AWAKE, STATE_DROWSY, STATE_ALARM,
)

# Durum -> (BGR renk, ekran etiketi).
_STATE_STYLE = {
    STATE_AWAKE:   ((0, 200, 0),     "UYANIK"),
    STATE_DROWSY:  ((0, 200, 255),   "YORGUN"),
    STATE_ALARM:   ((0, 0, 255),     "ALARM!"),
    STATE_NO_FACE: ((170, 170, 170), "YUZ YOK"),
}
_PANEL_BG = (28, 28, 30)   # Koyu panel zemini
_WHITE = (255, 255, 255)
_MUTED = (190, 190, 190)


def _blend_panel(frame, p1, p2, color, alpha):
    """Verilen dikdörtgene yarı saydam dolgu uygular (hologram tarzı)."""
    x1, y1 = p1
    x2, y2 = p2
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    overlay = roi.copy()
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)


def _draw_bar(frame, x, y, w, h, ratio, color, label):
    """Etiketli bir oran çubuğu çizer (ratio 0..1, taşma kırpılır)."""
    ratio = max(0.0, min(1.0, ratio))
    cv2.rectangle(frame, (x, y), (x + w, y + h), (70, 70, 70), 1, cv2.LINE_AA)
    fill = int(w * ratio)
    if fill > 0:
        cv2.rectangle(frame, (x, y), (x + fill, y + h), color, -1)
    cv2.putText(frame, label, (x, y - 4), FONT_UI, 0.4, _MUTED, 1, cv2.LINE_AA)


def draw_driver_hud(frame, state, blink_on=False):
    """
    Bir kare üzerine sürücü durumu HUD'unu çizer.

    frame    : BGR kare (yerinde değiştirilir)
    state    : drowsiness.DrowsinessState
    blink_on : ALARM'da kenar çerçevesinin "yanık" fazı (kare kare değiştirilir)
    """
    color, label = _STATE_STYLE.get(state.state, _STATE_STYLE[STATE_NO_FACE])

    # --- Sol üst durum paneli ---
    px, py = 14, 14
    pw, ph = 250, 150
    _blend_panel(frame, (px, py), (px + pw, py + ph), _PANEL_BG, 0.55)
    _round_rect(frame, (px, py), (px + pw, py + ph), color, 10, 1)
    _corner_brackets(frame, (px, py), (px + pw, py + ph), color, length=14, thick=2)

    # Sol renk vurgu şeridi + büyük durum etiketi.
    cv2.rectangle(frame, (px, py), (px + 6, py + ph), color, -1)
    cv2.putText(frame, label, (px + 18, py + 38),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2, cv2.LINE_AA)

    # Gerekçe satırı (ALARM/DROWSY nedeni).
    if state.reason:
        cv2.putText(frame, state.reason, (px + 18, py + 60),
                    FONT_UI, 0.42, _MUTED, 1, cv2.LINE_AA)

    # EAR ve PERCLOS çubukları.
    # EAR çubuğunu eşiğe göre ölçekle ki kapanma görsel olarak belirgin olsun.
    ear_ratio = state.ear / 0.4 if state.ear else 0.0
    _draw_bar(frame, px + 18, py + 82, pw - 36, 10, ear_ratio, color,
              f"EAR {state.ear:.2f}")
    _draw_bar(frame, px + 18, py + 112, pw - 36, 10, state.perclos,
              (0, 165, 255), f"PERCLOS {state.perclos * 100:.0f}%")

    # Esneme / baş düşmesi sayaçları (panel altında).
    cv2.putText(frame, f"Esneme: {state.yawns}   Bas dusmesi: {state.nods}",
                (px + 18, py + 140), FONT_UI, 0.42, _WHITE, 1, cv2.LINE_AA)

    # --- ALARM: kare kenarında yanıp sönen kırmızı çerçeve ---
    if state.alarm and blink_on:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 12, cv2.LINE_AA)
        msg = "!!! DIKKAT - UYAN !!!"
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 1.1, 3)
        cx = (w - tw) // 2
        _blend_panel(frame, (cx - 16, h - 80), (cx + tw + 16, h - 30),
                     (0, 0, 60), 0.5)
        cv2.putText(frame, msg, (cx, h - 44),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 0, 255), 3, cv2.LINE_AA)
