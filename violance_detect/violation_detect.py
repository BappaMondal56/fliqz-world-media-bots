import os
import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model

# -----------------------------
# Configuration
# -----------------------------
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 16
CLASSES_LIST = ["NonViolence", "Violence"]

# -----------------------------
# Model loading
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "MobBiLSTM_model_saved101.keras")

print(f"🧠 Loading violence detection model from: {MODEL_PATH}")
MoBiLSTM_model = load_model(MODEL_PATH)


# -----------------------------
# Video Evaluation
# -----------------------------
def evaluate_video_direct(video_path, violence_threshold=0.65, display=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    predicted_class_name = ''
    violence_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        resized_frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        normalized_frame = resized_frame.astype("float32") / 255.0
        frames_queue.append(normalized_frame)

        if len(frames_queue) == SEQUENCE_LENGTH:
            preds = MoBiLSTM_model.predict(np.expand_dims(frames_queue, axis=0), verbose=0)[0]
            predicted_label = np.argmax(preds)
            predicted_class_name = CLASSES_LIST[predicted_label]

            if predicted_class_name == "Violence":
                violence_frames += 1

    cap.release()
    violence_ratio = violence_frames / total_frames if total_frames > 0 else 0

    if violence_ratio >= violence_threshold:
        return "Violence", violence_ratio
    else:
        return "NonViolence", violence_ratio


# -----------------------------
# Image Evaluation
# -----------------------------
def predict_image(image_path, violence_threshold=0.70):
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Image not found: {image_path}")

    frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
    frame = frame.astype("float32") / 255.0

    # replicate same frame to match sequence model
    frames = np.array([frame] * SEQUENCE_LENGTH)
    frames = np.expand_dims(frames, axis=0)

    preds = MoBiLSTM_model.predict(frames, verbose=0)[0]
    violence_prob = float(preds[1])
    predicted_class_name = "Violence" if violence_prob >= violence_threshold else "NonViolence"

    print(f"\n🖼️ Image: {image_path}")
    print(f"🔍 Prediction: {predicted_class_name}")
    print(f"📊 Probabilities → NonViolence: {preds[0]:.4f}, Violence: {preds[1]:.4f}")

    return predicted_class_name, violence_prob


# -----------------------------
# Unified Entry Point
# -----------------------------
def predict_violation(file_path, file_type=None):
    """
    Detects violence in both images and videos.
    Returns:
        (label, probability)
    """
    ext = os.path.splitext(file_path)[-1].lower()
    video_exts = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]

    try:
        if ext in video_exts or (file_type and file_type.lower() == "videos"):
            label, prob = evaluate_video_direct(file_path, violence_threshold=0.65, display=False)
        else:
            label, prob = predict_image(file_path)
        return label, prob
    except Exception as e:
        print(f"❌ Error processing file '{file_path}': {e}")
        return "NonViolence", 0.0


def is_violence_detected(file_path, file_type=None, threshold=0.65):
    """
    Returns:
        True  -> Violence detected
        False -> No violence detected
    """
    label, prob = predict_violation(file_path, file_type)

    return bool(label == "Violence" and prob >= threshold)
