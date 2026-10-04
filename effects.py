"""
Particle effects like in the demo video: a cloud of glowing dots that
flies smoothly from one shape to another (text, heart, Saturn, cube).
"""
import math
import time

import cv2
import numpy as np

N = 1800
rng = np.random.default_rng(7)


# ---------- shape builders: each returns (points Nx3 in about -1..1, colors Nx3 BGR) ----------

def _text_points(text, color, n=N):
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), base = cv2.getTextSize(text, font, 3, 6)
    mask = np.zeros((th + base + 20, tw + 20), np.uint8)
    cv2.putText(mask, text, (10, th + 10), font, 3, 255, 6, cv2.LINE_AA)
    ys, xs = np.nonzero(mask)
    idx = rng.integers(0, len(xs), n)
    x = (xs[idx] - mask.shape[1] / 2) / (mask.shape[1] / 2)
    y = (ys[idx] - mask.shape[0] / 2) / (mask.shape[1] / 2)
    pts = np.stack([x, y, rng.normal(0, 0.01, n)], 1)
    return pts, np.tile(np.array(color, np.float32), (n, 1))


def _heart_points(n=N):
    t = rng.uniform(0, 2 * math.pi, n)
    x = 16 * np.sin(t) ** 3
    y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    pts = np.stack([x / 17, y / 17, rng.normal(0, 0.03, n)], 1)
    pts[:, :2] += rng.normal(0, 0.012, (n, 2))
    return pts, np.tile(np.array((60, 60, 255), np.float32), (n, 1))


def _saturn_points(n=N):

    n_ball = n // 3
    v = rng.normal(size=(n_ball, 3))
    v = 0.42 * v / np.linalg.norm(v, axis=1, keepdims=True)
    ang = rng.uniform(0, 2 * math.pi, n - n_ball)
    rad = rng.uniform(0.68, 1.0, n - n_ball)
    ring = np.stack([rad * np.cos(ang), rng.normal(0, 0.012, n - n_ball), rad * np.sin(ang)], 1)
    tilt = math.radians(25)
    rot = np.array([[math.cos(tilt), -math.sin(tilt), 0], [math.sin(tilt), math.cos(tilt), 0], [0, 0, 1]])
    ring = ring @ rot.T
    pts = np.vstack([v, ring])
    ball_col = np.tile(np.array((0, 140, 255), np.float32), (n_ball, 1))
    t = (rad - 0.68) / 0.32
    ring_col = np.stack([255 - 60 * t, 120 + 100 * t, 255 * (1 - t) + 120 * t], 1).astype(np.float32)
    return pts, np.vstack([ball_col, ring_col])


def _cube_points(n=N):
    pts = rng.uniform(-1, 1, (n, 3))
    axis = rng.integers(0, 3, n)
    pts[np.arange(n), axis] = rng.choice([-1, 1], n)
    pts *= 0.6
    return pts, np.tile(np.array((255, 170, 40), np.float32), (n, 1))


def build_shapes(name_text="MOHAMED"):
    return {
        "HI": _text_points("HI", (60, 255, 80)),
        "LOVE": _text_points("I LOVE YOU", (255, 220, 40)),
        "NAME": _text_points(name_text, (255, 60, 255)),
        "HEART": _heart_points(),
        "SATURN": _saturn_points(),
        "CUBE": _cube_points(),
    }


# ---------- the particle cloud ----------

class ParticleCloud:
    def __init__(self, name_text="MOHAMED"):
        self.shapes = build_shapes(name_text)
        self.pos = rng.normal(0, 0.8, (N, 3)).astype(np.float32)
        self.color = np.full((N, 3), 200, np.float32)
        self.target_pts = self.pos.copy()
        self.target_col = self.color.copy()
        self.speed = rng.uniform(3.5, 9.0, (N, 1)).astype(np.float32)  # each dot flies at its own pace
        self.name = None
        self.spin = 0.0
        self.alpha = 0.0
        self.alpha_target = 0.0
        self.last_t = time.time()

    def set_shape(self, name):
        """name = key in shapes, or None to let the dots drift away."""
        if name == self.name:
            return
        self.name = name
        if name is None:
            self.alpha_target = 0.0
            return
        pts, col = self.shapes[name]
        order = rng.permutation(N)
        self.target_pts, self.target_col = pts[order], col[order]
        self.alpha_target = 1.0

    def update(self, now=None):
        now = time.time() if now is None else now
        dt = max(0.0, min(now - self.last_t, 0.05))
        self.last_t = now
        self.spin += dt * (1.0 if self.name in ("SATURN", "CUBE") else 0.0)
        self.alpha += (self.alpha_target - self.alpha) * min(dt * 5, 1)

        goal = self.target_pts
        if self.name in ("SATURN", "CUBE"):
            c, s = math.cos(self.spin), math.sin(self.spin)
            rot_y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)
            goal = goal @ rot_y.T
        k = 1 - np.exp(-self.speed * dt)
        self.pos += (goal - self.pos) * k
        if self.name is None:
            self.pos += rng.normal(0, 0.01, self.pos.shape).astype(np.float32)
        self.color += (self.target_col - self.color) * min(dt * 6, 1)

    def draw(self, frame, center, size):
        """Draw the glowing dots on top of frame (in place)."""
        if self.alpha < 0.02:
            return
        h, w = frame.shape[:2]
        z = self.pos[:, 2]
        persp = 1.0 / (1.0 + 0.25 * z)
        xs = (center[0] + self.pos[:, 0] * size * persp).astype(int)
        ys = (center[1] + self.pos[:, 1] * size * persp).astype(int)
        ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        bright = np.clip(0.75 + 0.3 * (-z), 0.4, 1.1)[:, None]
        layer = np.zeros_like(frame)
        layer[ys[ok], xs[ok]] = np.clip(self.color[ok] * bright[ok], 0, 255).astype(np.uint8)
        layer = cv2.dilate(layer, np.ones((2, 2), np.uint8))
        glow = cv2.GaussianBlur(layer, (0, 0), 3)
        out = cv2.addWeighted(layer, 1.0, glow, 1.2, 0)
        a = self.alpha
        frame[:] = cv2.addWeighted(frame, 1 - 0.35 * a, np.zeros_like(frame), 0, 0)  # dim the camera a bit
        frame[:] = cv2.add(frame, (out * a).astype(np.uint8))
