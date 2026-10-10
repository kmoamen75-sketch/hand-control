"""Hand-landmark helpers (pixel conversion, distances, pinch)."""
import math


def px(hand, idx, w, h):
    lm = hand.landmark[idx]
    return lm.x * w, lm.y * h


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def hand_scale(hand, w, h):
    """How big the hand looks now (wrist -> middle knuckle). Used so every
    threshold works whether the hand is near or far from the camera."""
    return max(distance(px(hand, 0, w, h), px(hand, 9, w, h)), 1.0)


def pinch_ratio(hand, w, h):
    """~0 when thumb and index touch, ~1+ when fully spread."""
    return distance(px(hand, 4, w, h), px(hand, 8, w, h)) / hand_scale(hand, w, h)


class PinchDetector:
    """Pinch with hysteresis: it needs a clearly closed pinch to start and a
    clearly open hand to stop, so it doesn't flicker on/off at the edge."""

    def __init__(self, on=0.30, off=0.45):
        self.on, self.off = on, off
        self.active = False

    def update(self, hand, w, h):
        r = pinch_ratio(hand, w, h)
        if self.active and r > self.off:
            self.active = False
        elif not self.active and r < self.on:
            self.active = True
        return self.active

    def reset(self):
        self.active = False
