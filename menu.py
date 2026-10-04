"""
The menu lives ON your left hand: one button sits on each finger joint
(and one on the wrist). Open your left palm to see them, then hold your
right index fingertip over a button for a moment to pick it.
Buttons follow the hand smoothly and stay put even if tracking wobbles.
"""
import math
import time

import cv2

from utils import hand_scale, px

# button name -> hand landmark it sticks to
BUTTONS = {
    "CURSOR": 6,    # index middle-joint
    "DRAW": 10,     # middle finger joint
    "FX": 14,       # ring finger joint
    "VOLUME": 18,   # pinky joint
    "CLEAR": 0,     # wrist
}
DWELL_TIME = 0.6
FOLLOW = 0.45       # 0..1 how fast buttons chase the hand (lower = smoother)


class JointMenu:
    def __init__(self):
        self.pos = {}
        self.radius = 24
        self.hover = None
        self.hover_start = 0.0

    def _track(self, hand, w, h):
        scale = hand_scale(hand, w, h)
        self.radius = int(max(16, min(34, scale * 0.28)))
        for name, idx in BUTTONS.items():
            x, y = px(hand, idx, w, h)
            if name == "CLEAR":
                y += scale * 0.45
            if name in self.pos:
                ox, oy = self.pos[name]
                x, y = ox + (x - ox) * FOLLOW, oy + (y - oy) * FOLLOW
            self.pos[name] = (x, y)

    def update(self, hand, w, h, pointer, now=None):
        """Returns the name of a button that was just selected, or None."""
        now = time.time() if now is None else now
        self._track(hand, w, h)
        if pointer is None:
            self.hover = None
            return None
        best, best_d = None, self.radius * 1.25
        for name, (x, y) in self.pos.items():
            d = math.hypot(pointer[0] - x, pointer[1] - y)
            if d < best_d:
                best, best_d = name, d
        if best != self.hover:
            self.hover, self.hover_start = best, now
            return None
        if best is not None and now - self.hover_start >= DWELL_TIME:
            self.hover = None
            return best
        return None

    def reset(self):
        self.pos.clear()
        self.hover = None

    def draw(self, frame, active_mode, now=None):
        now = time.time() if now is None else now
        for name, (x, y) in self.pos.items():
            c = (int(x), int(y))
            on = name == self.hover
            current = name == active_mode
            color = (0, 255, 255) if on else ((0, 220, 0) if current else (230, 230, 230))
            overlay = frame.copy()
            cv2.circle(overlay, c, self.radius, color, -1)
            cv2.addWeighted(overlay, 0.35 if not on else 0.6, frame, 0.65 if not on else 0.4, 0, frame)
            cv2.circle(frame, c, self.radius, color, 2, cv2.LINE_AA)
            label = name
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.putText(frame, label, (c[0] - tw // 2, c[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20) if on else (255, 255, 255), 1, cv2.LINE_AA)
            if on:
                p = min((now - self.hover_start) / DWELL_TIME, 1.0)
                cv2.ellipse(frame, c, (self.radius + 6, self.radius + 6), -90, 0, 360 * p,
                            (0, 255, 0), 3, cv2.LINE_AA)


def draw_size_button(frame, point, ratio_0_to_1, min_radius=8, max_radius=60):
    """Brush-size circle on the left hand: spread thumb+index = bigger brush."""
    r = max(0.0, min(1.0, ratio_0_to_1))
    radius = int(r * (max_radius - min_radius) + min_radius)
    p = (int(point[0]), int(point[1]))
    cv2.circle(frame, p, radius, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, p, 3, (255, 0, 255), -1)
    return radius
