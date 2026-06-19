"""
utils/draw_utils.py
Helper functions for drawing bounding boxes, labels, and overlays on frames.
"""

import cv2
import numpy as np

# Color palette per class category
COLORS = {
    "person"    : (0,   200, 255),   # cyan
    "mask"      : (0,   255, 100),   # green
    "no_mask"   : (0,   0,   255),   # red
    "helmet"    : (0,   200, 255),   # cyan
    "no_helmet" : (0,   0,   255),   # red
    "gloves"    : (0,   255, 200),   # teal
    "no_gloves" : (0,   0,   255),   # red
    "vest"      : (255, 200, 0  ),   # gold
    "no_vest"   : (0,   0,   255),   # red
    "boots"     : (180, 100, 255),   # purple
    "no_boots"  : (0,   0,   255),   # red
    "default"   : (200, 200, 200),   # grey
}


def draw_box(frame, x1, y1, x2, y2, label, confidence, class_name=None):
    """Draw a bounding box with label on frame."""
    color = COLORS.get(class_name or label, COLORS["default"])
    # Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    # Background for text
    text = f"{label} {confidence:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def draw_id_box(frame, x1, y1, x2, y2, track_id, class_name="person"):
    """Draw a bounding box with a tracking ID label."""
    color = COLORS.get(class_name, COLORS["default"])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID:{track_id}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return frame


def draw_tripwire(frame, p1, p2, triggered=False):
    """Draw a horizontal tripwire line across the frame."""
    color = (0, 0, 255) if triggered else (0, 255, 255)
    thickness = 3 if triggered else 2
    cv2.line(frame, p1, p2, color, thickness)
    label = "!! TRIPWIRE BREACH !!" if triggered else "TRIPWIRE"
    cv2.putText(frame, label, (p1[0] + 10, p1[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255) if triggered else (0, 255, 255), 2, cv2.LINE_AA)
    return frame


def draw_tamper_alert(frame, reason):
    """Draw a full-frame tamper alert overlay."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 180), -1)
    alpha = 0.3
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.putText(frame, "!! CAMERA TAMPERED !!",
                (30, frame.shape[0] // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, reason,
                (30, frame.shape[0] // 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def draw_status_bar(frame, text, color=(0, 255, 0)):
    """Draw a status bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 28), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, text, (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return frame
