"""
Hand control -- smooth, camera-only (no eye tracking).

Modes (pick them from the menu on your LEFT hand)
-------------------------------------------------
CURSOR  Right hand moves the mouse. Pinch = press (hold + move = drag).
DRAW    Pinch with the right hand and move to draw.
FX      Make a shape with your hand and it turns into glowing particles:
          open hand      -> Saturn        fist          -> cube
          peace (2 fing) -> heart         3 fingers     -> your name
          1 finger       -> HI            "I love you"  -> I LOVE YOU
          two hands making a heart -> heart
VOLUME  Right hand above / below the middle line = volume up / down.

Open your LEFT palm to show the menu: a button sits on each finger joint.
Hold your right index fingertip on a button to choose it.
Press 'q' in the camera window to quit.
"""
import time

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

from effects import ParticleCloud
from gestures import StableGesture, classify, is_open_palm, two_hand_heart
from menu import JointMenu, draw_size_button
from smoothing import OneEuro2D
from utils import PinchDetector, pinch_ratio, px

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0            # no built-in delay after every call -> much smoother cursor

YOUR_NAME = "MOAMEN"           # shown by the 3-finger gesture in FX mode
CURSOR_LANDMARK = 6            # index MIDDLE JOINT: steadier than the fingertip, and it
                               # doesn't jump when you pinch (the tip moves, the joint doesn't)
ACTIVE_BOX = (0.18, 0.15, 0.82, 0.75)   # part of the camera image that maps to the WHOLE screen
CLICK_FREEZE = 0.25            # seconds the cursor holds still right after a pinch (clean clicks)
HAND_LOST_GRACE = 0.4          # seconds before we forget a hand that tracking briefly lost

SCREEN_W, SCREEN_H = pyautogui.size()
mp_hands = mp.solutions.hands


class Hand:
    """Keeps the last good landmarks for a short time so a one-frame
    tracking drop doesn't make everything flicker."""

    def __init__(self):
        self.lm = None
        self.seen = 0.0

    def set(self, lm, now):
        if lm is not None:
            self.lm, self.seen = lm, now
        elif now - self.seen > HAND_LOST_GRACE:
            self.lm = None
        return self.lm


def split_hands(results):
    out = {}
    if results.multi_hand_landmarks and results.multi_handedness:
        for lm, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
            out[hd.classification[0].label] = lm
    return out


def to_screen(point, w, h):
    x0, y0, x1, y1 = ACTIVE_BOX
    nx = (point[0] / w - x0) / (x1 - x0)
    ny = (point[1] / h - y0) / (y1 - y0)
    nx, ny = min(max(nx, 0.0), 1.0), min(max(ny, 0.0), 1.0)
    return nx * (SCREEN_W - 1), ny * (SCREEN_H - 1)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # always the newest frame -> less lag
    if not cap.isOpened():
        print("Couldn't open the camera. Try changing VideoCapture(0) to 1 or 2.")
        return

    right, left = Hand(), Hand()
    menu = JointMenu()
    cloud = ParticleCloud(YOUR_NAME)
    gesture = StableGesture(hold=0.25)
    pinch = PinchDetector()
    cursor_filter = OneEuro2D(min_cutoff=1.0, beta=0.05)
    draw_filter = OneEuro2D(min_cutoff=1.5, beta=0.08)
    canvas = None
    mode = "CURSOR"
    pinch_start = 0.0
    mouse_down = False
    prev_draw = None
    brush = 6
    last_volume = 0.0
    fx_anchor = None

    with mp_hands.Hands(
        max_num_hands=2, model_complexity=0,
        min_detection_confidence=0.6, min_tracking_confidence=0.6,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            if canvas is None:
                canvas = np.zeros_like(frame)
            now = time.time()

            found = split_hands(hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            R = right.set(found.get("Right"), now)
            L = left.set(found.get("Left"), now)

            # ---- menu on the left hand's joints ----
            menu_open = L is not None and is_open_palm(L, w, h)
            if menu_open:
                pointer = px(R, 8, w, h) if R is not None else None
                chosen = menu.update(L, w, h, pointer, now)
                if chosen == "CLEAR":
                    canvas[:] = 0
                elif chosen:
                    mode = chosen
                    gesture.update(None, now)
                    if mouse_down:
                        pyautogui.mouseUp()
                        mouse_down = False
                menu.draw(frame, mode, now)
            else:
                menu.reset()

            # ---- right hand drives the current mode (not while picking from the menu) ----
            acting = R is not None and not menu_open
            if not acting:
                pinch.reset()
                cursor_filter.reset()
                draw_filter.reset()
                prev_draw = None
                if mouse_down:
                    pyautogui.mouseUp()
                    mouse_down = False

            if acting and mode == "CURSOR":
                pinched = pinch.update(R, w, h)
                sx, sy = cursor_filter(to_screen(px(R, CURSOR_LANDMARK, w, h), w, h), now)
                if pinched and not mouse_down:
                    pinch_start = now
                    pyautogui.mouseDown()
                    mouse_down = True
                elif not pinched and mouse_down:
                    pyautogui.mouseUp()
                    mouse_down = False
                if not (mouse_down and now - pinch_start < CLICK_FREEZE):
                    pyautogui.moveTo(sx, sy, _pause=False)
                dot = px(R, CURSOR_LANDMARK, w, h)
                color = (0, 0, 255) if pinched else (0, 255, 0)
                cv2.circle(frame, (int(dot[0]), int(dot[1])), 10, color, -1, cv2.LINE_AA)
                bx0, by0, bx1, by1 = ACTIVE_BOX
                cv2.rectangle(frame, (int(bx0 * w), int(by0 * h)), (int(bx1 * w), int(by1 * h)),
                              (90, 90, 90), 1)

            elif acting and mode == "DRAW":
                pinched = pinch.update(R, w, h)
                fx, fy = draw_filter(px(R, 8, w, h), now)
                tip = (int(fx), int(fy))
                if pinched:
                    if prev_draw is not None:
                        cv2.line(canvas, prev_draw, tip, (0, 255, 255), brush, cv2.LINE_AA)
                    prev_draw = tip
                else:
                    prev_draw = None
                cv2.circle(frame, tip, max(4, brush // 2), (0, 255, 255), -1, cv2.LINE_AA)

            elif acting and mode == "VOLUME":
                y = px(R, 8, w, h)[1]
                mid = h / 2
                if now - last_volume > 0.12:
                    if y < mid - 40:
                        pyautogui.press("volumeup"); last_volume = now
                    elif y > mid + 40:
                        pyautogui.press("volumedown"); last_volume = now
                cv2.line(frame, (0, int(mid)), (w, int(mid)), (255, 255, 0), 1)
                cv2.circle(frame, (int(px(R, 8, w, h)[0]), int(y)), 10, (255, 255, 0), -1)

            # ---- FX: shapes made with the hand become particles ----
            if mode == "FX":
                shape = None
                if R is not None and L is not None and two_hand_heart(R, L, w, h):
                    seen = "HEART2"
                elif R is not None and not menu_open:
                    seen = classify(R, w, h)
                else:
                    seen = None
                stable = gesture.update(seen, now)
                shape = {"OPEN": "SATURN", "FIST": "CUBE", "PEACE": "HEART", "HEART2": "HEART",
                         "THREE": "NAME", "POINT": "HI", "LOVE": "LOVE"}.get(stable)
                hand_for_anchor = R if R is not None else L
                if hand_for_anchor is not None and shape is not None:
                    c = px(hand_for_anchor, 9, w, h)
                    if seen == "HEART2":
                        c = ((px(R, 8, w, h)[0] + px(L, 8, w, h)[0]) / 2,
                             (px(R, 8, w, h)[1] + px(L, 8, w, h)[1]) / 2)
                    target = (c[0], c[1] - 0.18 * h)
                    fx_anchor = target if fx_anchor is None else (
                        fx_anchor[0] + (target[0] - fx_anchor[0]) * 0.25,
                        fx_anchor[1] + (target[1] - fx_anchor[1]) * 0.25)
                cloud.set_shape(shape)
                cloud.update(now)
                if fx_anchor is not None:
                    cloud.draw(frame, fx_anchor, 0.28 * h)
            else:
                cloud.set_shape(None)
                cloud.update(now)

            # ---- brush-size button on the left hand while drawing ----
            if mode == "DRAW" and L is not None and not menu_open:
                ratio = (pinch_ratio(L, w, h) - 0.1) / 0.9
                brush = max(2, int(draw_size_button(frame, px(L, 9, w, h), ratio) / 4))

            # ---- compose ----
            display = frame
            mask = canvas.any(axis=2)
            display[mask] = canvas[mask]
            cv2.putText(display, f"Mode: {mode}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Hand Control", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if mouse_down:
        pyautogui.mouseUp()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
