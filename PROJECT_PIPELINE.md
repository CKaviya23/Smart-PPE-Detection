# Safety PPE Detection System — Full Pipeline

## Project Structure
```
safety_ppe_project/
├── app.py                        # Flask backend (main entry)
├── requirements.txt              # Python dependencies
├── config.py                     # Model paths & config
│
├── models/
│   └── model_loader.py          # Load Keras model once at startup
│
├── detectors/
│   ├── __init__.py
│   ├── person_detector.py       # Task 1: Person detection + tracking IDs
│   ├── ppe_detector.py          # Task 2: PPE item detection (mask/gloves/etc)
│   ├── tripwire_detector.py     # Task 3: Tripwire line crossing detection
│   └── tamper_detector.py       # Task 4: Camera tampering detection
│
├── utils/
│   ├── __init__.py
│   ├── draw_utils.py            # Bounding box / annotation drawing
│   └── frame_utils.py           # Frame preprocessing helpers
│
├── static/
│   ├── css/
│   │   └── style.css            # Frontend styling
│   ├── js/
│   │   └── main.js              # Webcam streaming + UI logic
│   └── uploads/                 # Temp uploaded images/videos
│
└── templates/
    └── index.html               # Single-page UI
```

## Pipeline Flow

```
[User Input] ──→ [Flask Route]
     │
     ├── /detect/photo    → decode image → run detector → return annotated image
     ├── /detect/video    → frame-by-frame → run detector → return video stream
     └── /detect/webcam   → MJPEG stream → run detector → stream annotated frames

[Detector Selection]
     │
     ├── Task Button: "Person"        → person_detector.py
     ├── Task Button: "Mask"          → ppe_detector.py  (class: mask)
     ├── Task Button: "Gloves"        → ppe_detector.py  (class: gloves)
     ├── Task Button: "Helmet"        → ppe_detector.py  (class: helmet)
     ├── Task Button: "Boots"         → ppe_detector.py  (class: boots)
     ├── Task Button: "Vest"          → ppe_detector.py  (class: vest)
     ├── Task Button: "Tripwire"      → tripwire_detector.py
     └── Task Button: "Tampering"     → tamper_detector.py

[Model]
     └── my_deep_learning_model.keras
           ↓
     Loaded ONCE at startup via model_loader.py
     Shared across all detectors

[Output]
     └── Annotated frame/image → displayed in right panel of UI
```

## Dataset Classes (inferred from model)
The model is expected to detect these PPE-related classes:
- person
- mask / no_mask
- helmet / no_helmet
- gloves / no_gloves
- vest / no_vest
- boots / no_boots

(Adjust class names in config.py to match your actual model output)
