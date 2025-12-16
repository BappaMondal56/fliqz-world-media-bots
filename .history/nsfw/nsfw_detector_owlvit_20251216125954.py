import torch
import cv2
import os
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from collections import defaultdict

# -----------------------------
# Load OWLv2 Model
# -----------------------------
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16")
model.eval()

# -----------------------------
# NSFW Labels
# -----------------------------
NSFW_LABELS = [
    "Porn",
    "Nudity",
    "Sexual Activity",
    "Genitalia",
    "Breasts",
    "Buttocks",
    "Explicit Content"
]

# -----------------------------
# Class-specific thresholds
# -----------------------------
CLASS_THRESHOLDS = {
    "Porn": 0.50,
    "Nudity": 0.50,
    "Sexual Activity": 0.50,
    "Genitalia": 0.50,
    "Breasts": 0.50,
    "Buttocks": 0.50,
    "Explicit Content": 0.50,
}

DEFAULT_THRESHOLD = 0.50

# -----------------------------
# Utility: IoU
# -----------------------------
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


# -----------------------------
# Remove overlapping boxes
# -----------------------------
def remove_overlaps(detections, iou_thresh=0.5):
    if not detections:
        return []

    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    kept = []

    for det in detections:
        if all(calculate_iou(det["box"], k["box"]) < iou_thresh for k in kept):
            kept.append(det)

    return kept


# -----------------------------
# Image NSFW Detection
# -----------------------------
def detect_nsfw_image(image_path):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        text=NSFW_LABELS,
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
        label_name = NSFW_LABELS[label]
        threshold = CLASS_THRESHOLDS.get(label_name, DEFAULT_THRESHOLD)

        if score >= threshold:
            detections.append({
                "label": label_name,
                "confidence": float(score),
                "box": [float(v) for v in box.tolist()]
            })

    detections = remove_overlaps(detections)
    return detections


# -----------------------------
# Video NSFW Detection
# -----------------------------
def detect_nsfw_video(video_path, skip_frames=20):
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    all_detections = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % skip_frames == 0:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            inputs = processor(
                text=NSFW_LABELS,
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
                label_name = NSFW_LABELS[label]
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


# -----------------------------
# Aggregate video detections
# -----------------------------
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


# -----------------------------
# Public API
# -----------------------------
def is_nsfw_detected(path):
    if not os.path.exists(path):
        return False

    ext = os.path.splitext(path)[1].lower()

    image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    video_exts = [".mp4", ".avi", ".mov", ".mkv"]

    if ext in image_exts:
        detections = detect_nsfw_image(path)
        return bool(detections)

    if ext in video_exts:
        raw = detect_nsfw_video(path)
        aggregated = aggregate_video_results(raw)
        return bool(aggregated)

    return False

result = is_nsfw_detected("storage/posts/images/nsfw4.jpg")
print("NSFW detected:", result)
