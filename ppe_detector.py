"""
detectors/ppe_detector.py
Task 2: PPE Detection
Detects specific PPE items (mask, helmet, gloves, vest, boots) per-button selection.
"""

import numpy as np
from models.model_loader import get_model
from config import PPE_CLASS_MAP, ALL_CLASSES, CONFIDENCE_THRESHOLD, NMS_THRESHOLD
from utils.frame_utils import preprocess_frame, decode_predictions, apply_nms, scale_boxes
from utils.draw_utils import draw_box, draw_status_bar


def detect_ppe(frame, ppe_type="mask"):
    """
    Detect a specific PPE item in the frame.

    Args:
        frame     : BGR numpy array
        ppe_type  : one of "mask", "helmet", "gloves", "vest", "boots"
                    (maps to button clicked in the UI)

    Returns:
        annotated_frame : frame with boxes drawn
        results         : list of {class_name, confidence, x1, y1, x2, y2}
    """
    model = get_model()
    blob, sx, sy = preprocess_frame(frame)

    raw = model.predict(blob, verbose=0)
    detections = decode_predictions(raw, CONFIDENCE_THRESHOLD)
    detections = apply_nms(detections, NMS_THRESHOLD)
    detections = scale_boxes(detections, sx, sy)

    # Get class IDs relevant to this PPE type
    target_classes = PPE_CLASS_MAP.get(ppe_type, [])
    target_ids = {ALL_CLASSES.index(c): c for c in target_classes if c in ALL_CLASSES}

    results = []
    compliant_count = 0
    violation_count = 0

    for det in detections:
        if det["class_id"] not in target_ids:
            continue

        class_name = target_ids[det["class_id"]]
        label = class_name.replace("_", " ").upper()

        draw_box(frame,
                 det["x1"], det["y1"], det["x2"], det["y2"],
                 label=label,
                 confidence=det["confidence"],
                 class_name=class_name)

        results.append({
            "class_name" : class_name,
            "confidence" : det["confidence"],
            "x1": det["x1"], "y1": det["y1"],
            "x2": det["x2"], "y2": det["y2"]
        })

        # Compliance logic: "no_X" = violation
        if class_name.startswith("no_"):
            violation_count += 1
        else:
            compliant_count += 1

    status = (f"{ppe_type.upper()} — "
              f"Compliant: {compliant_count}  Violations: {violation_count}")
    color = (0, 255, 0) if violation_count == 0 else (0, 0, 255)
    draw_status_bar(frame, status, color=color)

    return frame, results


def detect_all_ppe(frame):
    """
    Detect ALL PPE classes in one pass (used for the 'All Safety' button).
    Returns annotated frame and grouped results.
    """
    model = get_model()
    blob, sx, sy = preprocess_frame(frame)

    raw = model.predict(blob, verbose=0)
    detections = decode_predictions(raw, CONFIDENCE_THRESHOLD)
    detections = apply_nms(detections, NMS_THRESHOLD)
    detections = scale_boxes(detections, sx, sy)

    all_class_map = {ALL_CLASSES.index(c): c for c in ALL_CLASSES}
    results = []
    violations = 0

    for det in detections:
        class_name = all_class_map.get(det["class_id"], "unknown")
        if class_name == "unknown":
            continue
        label = class_name.replace("_", " ").upper()
        draw_box(frame,
                 det["x1"], det["y1"], det["x2"], det["y2"],
                 label=label,
                 confidence=det["confidence"],
                 class_name=class_name)
        results.append({"class_name": class_name, "confidence": det["confidence"]})
        if class_name.startswith("no_"):
            violations += 1

    draw_status_bar(frame,
                    f"All PPE — Total: {len(results)}  Violations: {violations}",
                    color=(0, 0, 255) if violations else (0, 255, 0))
    return frame, results
