"""
detectors/tamper_detector.py
Task 4: Camera Tampering Detection
Detects: blur (lens covered), brightness change (blocked/blinded),
and sudden scene change (camera moved/rotated).
"""

import cv2
import numpy as np
from config import (TAMPER_BLUR_THRESHOLD, TAMPER_BRIGHTNESS_LOW,
                    TAMPER_BRIGHTNESS_HIGH, TAMPER_MOTION_THRESHOLD)
from utils.draw_utils import draw_tamper_alert, draw_status_bar

_prev_frame_gray = None   # Reference frame for scene-change detection


def reset_tamper():
    global _prev_frame_gray
    _prev_frame_gray = None


def _blur_score(gray):
    """Laplacian variance — lower = blurrier (lens covered)."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _brightness(gray):
    """Mean pixel brightness."""
    return float(np.mean(gray))


def _scene_change_score(gray):
    """Mean absolute diff from reference frame."""
    global _prev_frame_gray
    if _prev_frame_gray is None or _prev_frame_gray.shape != gray.shape:
        _prev_frame_gray = gray.copy()
        return 0.0
    score = float(np.mean(np.abs(gray.astype(np.float32) -
                                 _prev_frame_gray.astype(np.float32))))
    _prev_frame_gray = gray.copy()
    return score


def detect_tampering(frame, reset=False):
    """
    Analyse a frame for camera tampering signs.

    Returns:
        annotated_frame : frame with alert overlay if tampered
        results         : dict {tampered, reason, blur, brightness, scene_change}
    """
    if reset:
        reset_tamper()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur      = _blur_score(gray)
    bright    = _brightness(gray)
    scene_chg = _scene_change_score(gray)

    tampered = False
    reasons  = []

    if blur < TAMPER_BLUR_THRESHOLD:
        tampered = True
        reasons.append(f"Blur detected ({blur:.1f})")

    if bright < TAMPER_BRIGHTNESS_LOW:
        tampered = True
        reasons.append(f"Camera blocked (brightness {bright:.1f})")

    if bright > TAMPER_BRIGHTNESS_HIGH:
        tampered = True
        reasons.append(f"Camera blinded (brightness {bright:.1f})")

    if scene_chg > TAMPER_MOTION_THRESHOLD:
        tampered = True
        reasons.append(f"Scene change ({scene_chg:.1f})")

    reason_str = " | ".join(reasons) if reasons else "None"

    if tampered:
        frame = draw_tamper_alert(frame, reason_str)
    else:
        status = (f"Camera OK — Blur:{blur:.0f}  Bright:{bright:.0f}  "
                  f"Change:{scene_chg:.1f}")
        draw_status_bar(frame, status, color=(0, 255, 0))

    return frame, {
        "tampered"     : tampered,
        "reason"       : reason_str,
        "blur"         : round(blur, 2),
        "brightness"   : round(bright, 2),
        "scene_change" : round(scene_chg, 2),
    }
