"""
Recognise hand shapes. Finger "up" is judged by distance from the wrist,
so it works even if the hand is tilted or sideways.
"""
from utils import distance, px, hand_scale

# (tip, pip) for index, middle, ring, pinky
_FINGERS = [(8, 6), (12, 10), (16, 14), (20, 18)]


def finger_states(hand, w, h):
    """Returns (thumb, index, middle, ring, pinky) as 0/1."""
    wrist = px(hand, 0, w, h)
    states = []
    # thumb: tip farther from the pinky base than the thumb's own base joint
    pinky_base = px(hand, 17, w, h)
    thumb_out = distance(px(hand, 4, w, h), pinky_base) > 1.15 * distance(px(hand, 2, w, h), pinky_base)
    states.append(int(thumb_out))
    for tip, pip in _FINGERS:
        up = distance(px(hand, tip, w, h), wrist) > 1.15 * distance(px(hand, pip, w, h), wrist)
        states.append(int(up))
    return tuple(states)


def is_open_palm(hand, w, h):
    return all(finger_states(hand, w, h)[1:])


def classify(hand, w, h):
    """OPEN, FIST, POINT, PEACE, THREE, LOVE or None."""
    t, i, m, r, p = finger_states(hand, w, h)
    if i and p and not m and not r:
        return "LOVE"                  # thumb + index + pinky ("I love you" sign)
    if i and m and r and p:
        return "OPEN"
    if not (i or m or r or p):
        return "FIST"
    if i and m and r and not p:
        return "THREE"
    if i and m and not r and not p:
        return "PEACE"
    if i and not (m or r or p):
        return "POINT"
    return None


def two_hand_heart(right, left, w, h):
    """Both index tips touching AND both thumb tips touching = a heart."""
    scale = (hand_scale(right, w, h) + hand_scale(left, w, h)) / 2
    index_gap = distance(px(right, 8, w, h), px(left, 8, w, h))
    thumb_gap = distance(px(right, 4, w, h), px(left, 4, w, h))
    return index_gap < 0.9 * scale and thumb_gap < 1.1 * scale


class StableGesture:
    """A gesture only 'counts' after it was held for a moment, so a hand that
    is passing through a shape doesn't flash random effects."""

    def __init__(self, hold=0.25):
        self.hold = hold
        self.candidate = None
        self.since = 0.0
        self.value = None

    def update(self, name, now):
        if name != self.candidate:
            self.candidate, self.since = name, now
        if self.candidate is not None and now - self.since >= self.hold:
            self.value = self.candidate
        elif self.candidate is None:
            self.value = None
        return self.value
