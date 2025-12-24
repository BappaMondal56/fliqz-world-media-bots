import torch
import cv2
import os
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from collections import defaultdict

# =========================================================
# Load OWLv2
# =========================================================
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16")
model.eval()

# =========================================================
# ANIMAL LABELS
# =========================================================
ANIMAL_LABELS = [
    "Dog",
    "Cat",
    "Cow",
    "Horse",
    "Goat",
    "Sheep",
    "Pig",
    "Elephant",
    "Tiger",
    "Lion",
    "Bear",
    "Deer",
    "Monkey",
    "Bird",
    "Snake"
]

# =========================================================
# Thresholds (ALL SET TO 0.5)
# =========================================================
CLASS_THRESHOLDS = {
    "Dog": 0.50,
    "Cat": 0.50,
    "Cow": 0.50,
    "Horse": 0.50,
    "Goat": 0.50,
    "Sheep": 0.50,
    "Pig": 0.50,
    "Elephant": 0.50,
    "Tiger": 0.50,
    "Lion": 0.50,
    "Bear": 0.50,
    "Deer": 0.50,
    "Monkey": 0.50,
    "Bird": 0.50,
    "Snake": 0.50
}

DEFAULT_THRESHOLD = 0.50

# =========================================================
# IoU utility (used only for VIDEO)
# =========================================================
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def remove_overlaps(detections, iou_thresh=0.5):
    if not detections:
        return []

    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    kept = []

    for det in detections:
        if all(calculate_iou(det["box"], k["box"]) < iou_thresh for k in kept):
            kept.append(det)

    return kept

# =========================================================
# IMAGE ANIMAL DETECTION
# =========================================================
def detect_animal_image(image_path):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        text=ANIMAL_LABELS,
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=0.25
    )[0]

    detections = []

    for score, label, box in zip(
        results["scores"],
        results["labels"],
        results["boxes"]
    ):
        label_name = ANIMAL_LABELS[label]
        threshold = CLASS_THRESHOLDS.get(label_name, DEFAULT_THRESHOLD)

        if score >= threshold:
            detections.append({
                "label": label_name,
                "confidence": float(score),
                "box": [float(v) for v in box.tolist()]
            })

    return detections

# =========================================================
# VIDEO ANIMAL DETECTION
# =========================================================
def detect_animal_video(video_path, skip_frames=20):
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    all_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % skip_frames == 0:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            inputs = processor(
                text=ANIMAL_LABELS,
                images=pil_img,
                return_tensors="pt"
            )

            with torch.no_grad():
                outputs = model(**inputs)

            target_sizes = torch.tensor([pil_img.size[::-1]])
            results = processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=0.25
            )[0]

            frame_dets = []
            for score, label, box in zip(
                results["scores"],
                results["labels"],
                results["boxes"]
            ):
                label_name = ANIMAL_LABELS[label]
                threshold = CLASS_THRESHOLDS.get(label_name, DEFAULT_THRESHOLD)

                if score >= threshold:
                    frame_dets.append({
                        "frame": frame_id,
                        "label": label_name,
                        "confidence": float(score),
                        "box": [float(v) for v in box.tolist()]
                    })

            frame_dets = remove_overlaps(frame_dets)
            all_detections.extend(frame_dets)

        frame_id += 1

    cap.release()
    return all_detections

# =========================================================
# VIDEO AGGREGATION
# =========================================================
def aggregate_video_results(detections, min_frames=3):
    label_frames = defaultdict(list)

    for det in detections:
        label_frames[det["label"]].append(det["confidence"])

    aggregated = {}
    for label, confs in label_frames.items():
        if len(confs) >= min_frames:
            aggregated[label] = {
                "frames": len(confs),
                "avg_conf": round(sum(confs) / len(confs), 3),
                "max_conf": round(max(confs), 3)
            }

    return aggregated

# =========================================================
# PUBLIC API
# =========================================================
def is_animal_detected(path):
    if not os.path.exists(path):
        return False

    ext = os.path.splitext(path)[1].lower()

    image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    video_exts = [".mp4", ".avi", ".mov", ".mkv"]

    # ---------- IMAGE ----------
    if ext in image_exts:
        detections = detect_animal_image(path)
        return any(det["confidence"] >= 0.50 for det in detections)

    # ---------- VIDEO ----------
    if ext in video_exts:
        raw = detect_animal_video(path)
        aggregated = aggregate_video_results(raw)
        return bool(aggregated)

    return False
