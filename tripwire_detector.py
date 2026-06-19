"""
detectors/tripwire_detector.py
Task 3: Tripwire / Line-Crossing Detection
Detects persons crossing a configurable horizontal line.
"""

import numpy as np
import cv2
from collections import defaultdict

from models.model_loader import get_model
from config import (PERSON_CLASSES, ALL_CLASSES, TRIPWIRE_LINE,
                    CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
from utils.frame_utils import preprocess_frame, decode_predictions, apply_nms, scale_boxes
from utils.draw_utils import draw_tripwire, draw_id_box, draw_status_bar

# Track previous centroid Y-position to detect crossing direction
_prev_positions = {}   # person_index → previous_cy


def reset_tripwire():
    global _prev_positions
    _prev_positions = {}


def _get_tripwire_points(frame_h, frame_w):
    """Convert relative (0–1) tripwire coords to pixel coords."""
    x1 = int(TRIPWIRE_LINE["start"][0] * frame_w)
    y1 = int(TRIPWIRE_LINE["start"][1] * frame_h)
    x2 = int(TRIPWIRE_LINE["end"][0]   * frame_w)
    y2 = int(TRIPWIRE_LINE["end"][1]   * frame_h)
    return (x1, y1), (x2, y2)


def _is_crossing(cy_prev, cy_curr, line_y):
    """Check if centroid crossed the line between two frames."""
    return (cy_prev < line_y <= cy_curr) or (cy_curr < line_y <= cy_prev)


def detect_tripwire(frame, reset=False):
    """
    Draw a tripwire line and alert when a person crosses it.

    Args:
        frame  : BGR numpy array
        reset  : reset crossing history (new video/image)

    Returns:
        annotated_frame : frame with tripwire and any alerts drawn
        results         : dict {total_persons, breach_count, breached_ids}
    """
    global _prev_positions
    if reset:
        reset_tripwire()

    h, w = frame.shape[:2]
    p1, p2 = _get_tripwire_points(h, w)
    line_y = p1[1]   # horizontal line Y

    model = get_model()
    blob, sx, sy = preprocess_frame(frame)
    raw = model.predict(blob, verbose=0)
    detections = decode_predictions(raw, CONFIDENCE_THRESHOLD)
    detections = apply_nms(detections, NMS_THRESHOLD)
    detections = scale_boxes(detections, sx, sy)

    person_ids = [ALL_CLASSES.index(c) for c in PERSON_CLASSES if c in ALL_CLASSES]
    persons = [d for d in detections if d["class_id"] in person_ids]

    breach_count = 0
    breached = []

    for i, det in enumerate(persons):
        cx = (det["x1"] + det["x2"]) // 2
        cy = (det["y1"] + det["y2"]) // 2

        triggered = False
        if i in _prev_positions:
            if _is_crossing(_prev_positions[i], cy, line_y):
                triggered = True
                breach_count += 1
                breached.append(i)

        _prev_positions[i] = cy
        draw_id_box(frame, det["x1"], det["y1"], det["x2"], det["y2"],
                    track_id=i, class_name="person")

    any_breach = breach_count > 0
    draw_tripwire(frame, p1, p2, triggered=any_breach)

    status = (f"TRIPWIRE BREACH! Count: {breach_count}" if any_breach
              else f"Tripwire OK — Persons: {len(persons)}")
    draw_status_bar(frame, status, color=(0, 0, 255) if any_breach else (0, 255, 0))

    return frame, {
        "total_persons": len(persons),
        "breach_count": breach_count,
        "breached_ids": breached
    }
