"""
person_detector.py
Task 1: Person Detection with Tracking IDs
Uses YOLOv8n pretrained COCO model (class 0 = person)
Auto-downloads on first run.
"""

import numpy as np
import cv2
from collections import OrderedDict
from ultralytics import YOLO

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
CONF_THRESHOLD  = 0.30
PERSON_CLASS_ID = 0    # COCO class 0 = person

_model = None

def get_person_model():
    global _model
    if _model is None:
        print("[PersonDetector] Loading yolov8n.pt (COCO)...")
        _model = YOLO("yolov8n.pt")  # auto-downloads ~6MB
        print("[PersonDetector] Ready ✓")
    return _model

# ─────────────────────────────────────────────
#  Centroid Tracker
# ─────────────────────────────────────────────
class CentroidTracker:
    def __init__(self, max_disappeared=40):
        self.next_id     = 0
        self.objects     = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_id]     = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, obj_id):
        del self.objects[obj_id]
        del self.disappeared[obj_id]

    def update(self, rects):
        if len(rects) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        input_centroids = np.array([
            ((x1+x2)//2, (y1+y2)//2) for x1,y1,x2,y2 in rects
        ])

        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(tuple(c))
        else:
            obj_ids       = list(self.objects.keys())
            obj_centroids = list(self.objects.values())
            D = np.linalg.norm(
                np.array(obj_centroids)[:,None] - input_centroids[None,:],
                axis=2
            )
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                obj_id = obj_ids[row]
                self.objects[obj_id]     = tuple(input_centroids[col])
                self.disappeared[obj_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            for row in set(range(D.shape[0])) - used_rows:
                self.disappeared[obj_ids[row]] += 1
                if self.disappeared[obj_ids[row]] > self.max_disappeared:
                    self.deregister(obj_ids[row])

            for col in set(range(len(input_centroids))) - used_cols:
                self.register(tuple(input_centroids[col]))

        return self.objects


_tracker = CentroidTracker(max_disappeared=40)

def reset_tracker():
    global _tracker
    _tracker = CentroidTracker(max_disappeared=40)

# ─────────────────────────────────────────────
#  Draw
# ─────────────────────────────────────────────
def draw_id_box(frame, x1, y1, x2, y2, track_id, conf):
    colour = (255, 200, 0)
    cv2.rectangle(frame, (x1,y1), (x2,y2), colour, 2)
    label = f"Person ID:{track_id}  {conf:.0%}"
    (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+4, y1), colour, -1)
    cv2.putText(frame, label, (x1+2, y1-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)

def draw_status_bar(frame, text, colour=(0,220,60)):
    h,w = frame.shape[:2]
    cv2.rectangle(frame, (0,h-30), (w,h), (15,15,15), -1)
    cv2.putText(frame, text, (8,h-9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)

# ─────────────────────────────────────────────
#  Main Detection Function
# ─────────────────────────────────────────────
def detect_persons(frame, reset=False):
    if reset:
        reset_tracker()

    model = get_person_model()

    results_yolo = model(
        frame,
        conf=CONF_THRESHOLD,
        iou=0.45,
        classes=[PERSON_CLASS_ID],
        verbose=False
    )

    person_dets = []
    for r in results_yolo:
        for box in r.boxes:
            if int(box.cls[0]) != PERSON_CLASS_ID:
                continue
            conf = float(box.conf[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            person_dets.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,"conf":conf})

    rects   = [(d["x1"],d["y1"],d["x2"],d["y2"]) for d in person_dets]
    tracked = _tracker.update(rects)

    final_results = []
    for i, det in enumerate(person_dets):
        cx = (det["x1"]+det["x2"])//2
        cy = (det["y1"]+det["y2"])//2
        best_id=i; best_dist=float("inf")
        for tid, centroid in tracked.items():
            d = abs(centroid[0]-cx)+abs(centroid[1]-cy)
            if d < best_dist:
                best_dist=d; best_id=tid

        draw_id_box(frame, det["x1"],det["y1"],det["x2"],det["y2"],
                    track_id=best_id, conf=det["conf"])
        final_results.append({
            "id"        : int(best_id),
            "x1"        : det["x1"], "y1": det["y1"],
            "x2"        : det["x2"], "y2": det["y2"],
            "confidence": round(det["conf"],3)
        })

    draw_status_bar(frame, f"Persons Detected: {len(final_results)}", colour=(0,200,255))
    return frame, final_results