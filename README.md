# Safety PPE Detection

Lightweight PPE (Personal Protective Equipment) detection and monitoring utilities built with YOLO/ PyTorch models. The repo includes live detection, detectors for tampering and tripwire events, utilities for model loading and drawing, and training dataset artifacts.

Files of interest
- Project entry / demo: [app.py](app.py)
- Core detectors: [ppe_detector.py](ppe_detector.py), [person_detector.py](person_detector.py), [tamper_detector.py](tamper_detector.py), [tripwire_detector.py](tripwire_detector.py)
- Model helpers: [model_loader.py](model_loader.py), [ppe_detection.pt](ppe_detection.pt), [ppe_model.pt](ppe_model.pt), [ppe_full.pt](ppe_full.pt)
- Utilities: [draw_utils.py](draw_utils.py), [frame_utils.py](frame_utils.py)
- Dataset & training: [dataset/](dataset/) (see [dataset/data.yaml](dataset/data.yaml) and [dataset/trainner.py](dataset/trainner.py))
- Requirements: [requirements.txt](requirements.txt)

Quick start

1. Create and activate a Python virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the app (demo / live detection):

```powershell
python app.py
```

3. Run an individual detector (example):

```powershell
python ppe_detector.py --source path/to/video_or_image
```

Notes
- Model weight files (*.pt) are included in the repo root and under `ppe_yolov8s_run/weights/`. These are large binary artifacts — they are ignored by the repository's `.gitignore` if you add additional local/temporary models.
- The dataset folder contains training assets and runs; see [dataset/trainner.py](dataset/trainner.py) for the existing training helper.
- If you want to train or fine-tune new models, consider using the Ultralytics/YOLOv8 training commands and point `data=` to [dataset/data.yaml](dataset/data.yaml).

Troubleshooting
- If the app fails to find GPU drivers, ensure CUDA and the correct PyTorch build are installed for your CUDA toolkit.
- Check `requirements.txt` for package versions; use a clean virtual environment to avoid conflicts.

Contributing
- Open issues or pull requests for bug fixes or improvements. Add tests in `tests/` when possible.

License
- Add a license file if you plan to publish this project; otherwise assume internal use.
