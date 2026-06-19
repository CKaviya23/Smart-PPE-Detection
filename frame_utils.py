"""
utils/frame_utils.py
Frame preprocessing helpers shared across all detectors.
"""

import cv2
import numpy as np
import base64
from config import INPUT_SIZE


def preprocess_frame(frame, input_size=INPUT_SIZE):
    """
    Resize and normalize a frame for model inference.
    Returns:
        blob: float32 array shape (1, H, W, 3), values in [0, 1]
        scale_x: width scale factor (original_w / input_w)
        scale_y: height scale factor (original_h / input_h)
    """
    original_h, original_w = frame.shape[:2]
    resized = cv2.resize(frame, (input_size[1], input_size[0]))
    blob = resized.astype(np.float32) / 255.0
    blob = np.expand_dims(blob, axis=0)   # (1, H, W, 3)

    scale_x = original_w / input_size[1]
    scale_y = original_h / input_size[0]
    return blob, scale_x, scale_y


def decode_predictions(raw_output, confidence_threshold=0.5):
    """
    Decode raw model output into detections.
    Assumes output shape (1, num_boxes, 5 + num_classes)
    where last dims = [x_center, y_center, w, h, obj_conf, class0, class1, ...]
    
    Returns list of dicts: {x1, y1, x2, y2, confidence, class_id}
    """
    detections = []
    output = raw_output[0]   # (num_boxes, 5+C)

    for det in output:
        obj_conf = float(det[4])
        if obj_conf < confidence_threshold:
            continue
        class_scores = det[5:]
        class_id = int(np.argmax(class_scores))
        class_conf = float(class_scores[class_id])
        confidence = obj_conf * class_conf
        if confidence < confidence_threshold:
            continue

        cx, cy, w, h = det[0], det[1], det[2], det[3]
        x1 = int((cx - w / 2))
        y1 = int((cy - h / 2))
        x2 = int((cx + w / 2))
        y2 = int((cy + h / 2))
        detections.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "confidence": confidence,
            "class_id": class_id
        })
    return detections


def apply_nms(detections, nms_threshold=0.4):
    """Apply Non-Maximum Suppression to remove overlapping boxes."""
    if not detections:
        return []
    boxes = [[d["x1"], d["y1"], d["x2"] - d["x1"], d["y2"] - d["y1"]] for d in detections]
    scores = [d["confidence"] for d in detections]
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.0, nms_threshold)
    if len(indices) == 0:
        return []
    return [detections[i] for i in indices.flatten()]


def frame_to_base64_jpeg(frame, quality=85):
    """Encode a numpy frame to base64 JPEG string for JSON response."""
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode("utf-8")


def bytes_to_frame(image_bytes):
    """Convert uploaded image bytes to numpy BGR frame."""
    arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def scale_boxes(detections, scale_x, scale_y):
    """Scale bounding boxes back to original frame dimensions."""
    for d in detections:
        d["x1"] = int(d["x1"] * scale_x)
        d["y1"] = int(d["y1"] * scale_y)
        d["x2"] = int(d["x2"] * scale_x)
        d["y2"] = int(d["y2"] * scale_y)
    return detections
