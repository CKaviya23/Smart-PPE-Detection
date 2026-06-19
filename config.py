import os

# ─────────────────────────────────────────────
#  Model Configuration
# ─────────────────────────────────────────────
MODEL_PATH = r"C:\Users\kaviya-intern\Downloads\my_deep_learning_model.keras"

# ─────────────────────────────────────────────
#  Dataset Classes
#  Adjust these to match your actual model's class output order
# ─────────────────────────────────────────────
ALL_CLASSES = [
    "person",
    "mask",
    "no_mask",
    "helmet",
    "no_helmet",
    "gloves",
    "no_gloves",
    "vest",
    "no_vest",
    "boots",
    "no_boots",
]

# Which classes belong to which task
PERSON_CLASSES     = ["person"]
PPE_CLASS_MAP = {
    "mask"    : ["mask", "no_mask"],
    "helmet"  : ["helmet", "no_helmet"],
    "gloves"  : ["gloves", "no_gloves"],
    "vest"    : ["vest", "no_vest"],
    "boots"   : ["boots", "no_boots"],
}

# ─────────────────────────────────────────────
#  Detection Thresholds
# ─────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD        = 0.4
INPUT_SIZE           = (416, 416)   # Resize input frame to this before inference

# ─────────────────────────────────────────────
#  Tripwire Configuration
# ─────────────────────────────────────────────
# Line defined as two points: (x1,y1) → (x2,y2)  in % of frame size (0.0–1.0)
TRIPWIRE_LINE = {
    "start": (0.0, 0.5),   # left-center
    "end"  : (1.0, 0.5),   # right-center
}

# ─────────────────────────────────────────────
#  Camera Tampering Configuration
# ─────────────────────────────────────────────
TAMPER_BLUR_THRESHOLD       = 80.0   # Laplacian variance below this = blurred/covered
TAMPER_BRIGHTNESS_LOW       = 20     # Mean brightness below = covered
TAMPER_BRIGHTNESS_HIGH      = 240    # Mean brightness above = overexposed
TAMPER_MOTION_THRESHOLD     = 50.0   # Sudden scene change threshold

# ─────────────────────────────────────────────
#  Upload Folder
# ─────────────────────────────────────────────
UPLOAD_FOLDER = os.path.join("static", "uploads")
ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg", "bmp", "webp"}
ALLOWED_VIDEO_EXT = {"mp4", "avi", "mov", "mkv"}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024   # 100 MB max upload
