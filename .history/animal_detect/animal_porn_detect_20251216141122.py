from ultralytics import YOLO
import os
import cv2

# --------------------------------------------------
# Load YOLOv8 model
# --------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(SCRIPT_DIR, "yolov8n.pt")
model = YOLO(model_path)

ANIMAL_CLASSES = {
    "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra",
    "giraffe", "mouse"
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


# --------------------------------------------------
# Image detection
# --------------------------------------------------
def detect_animal_image(image_path):
    results = model(image_path)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if label in ANIMAL_CLASSES:
                return True, label

    return False, None


# --------------------------------------------------
# Video detection (frame sampling)
# --------------------------------------------------
def detect_animal_video(video_path, frame_skip=15):
    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample frames (performance-friendly)
        if frame_id % frame_skip == 0:
            results = model(frame, verbose=False)

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    if label in ANIMAL_CLASSES:
                        cap.release()
                        return True, label

        frame_id += 1

    cap.release()
    return False, None


# --------------------------------------------------
# Public API (image + video)
# --------------------------------------------------
def has_animal(path):
    if not os.path.exists(path):
        return False, None

    ext = os.path.splitext(path)[1].lower()

    if ext in IMAGE_EXTS:
        return detect_animal_image(path)

    if ext in VIDEO_EXTS:
        return detect_animal_video(path)

    return False, None
