import os
import sys
import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from nsfw.nsfw_detector import NSFWDetector
from animal_detect.animal_porn_detect import has_animal

# -----------------------------
#  Fix for module import
# -----------------------------
# Add parent directory to sys.path so meetup_detect can be imported
BASE_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_PARENT_DIR not in sys.path:
    sys.path.append(BASE_PARENT_DIR)

from meetup_detect.personal_details_detect import extract_text_from_file, isPersonalDetails

# -----------------------------
#  Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "age_detection_model.h5")

# -----------------------------
#  Load Models
# -----------------------------
print("🧠 Loading models...")

try:
    model = load_model(MODEL_PATH)
    print(f"✅ Loaded TensorFlow Age Detection model: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Failed to load TensorFlow age model: {e}")
    model = None

nsfw_detector = NSFWDetector()

# -----------------------------
#  Age / Minor Detection
# -----------------------------
def predict_age_group(image_path, img_size=(128, 128)):
    """
    Predict if the person is a minor (<18) or adult (18+).
    Returns True if minor, False if adult.
    """
    if model is None:
        return False

    if not os.path.exists(image_path):
        raise ValueError(f"File not found: {image_path}")

    img = load_img(image_path, target_size=img_size)
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)[0][0]

    if pred > 1.5:
        return pred < 18  # True = Minor
    else:
        return pred <= 0.5  # fallback probability-based decision

def detect_minor(image_path):
    """Run age detection on image."""
    return predict_age_group(image_path)

# -----------------------------
#  NSFW + Animal Detection
# -----------------------------
def detect_nsfw(image_path):
    """Detect NSFW content using NSFWDetector."""
    result = nsfw_detector.predict(image_path)
    return result.get("is_nsfw", False)

def detect_animal(image_path):
    """Detect animal presence (for animal-related NSFW)."""
    found, _ = has_animal(image_path)
    return found

# -----------------------------
#  Personal Details Detection
# -----------------------------
def detect_personal_details(image_path):
    """Detect personal information in image using OCR + NLP."""
    text = extract_text_from_file(image_path)
    return isPersonalDetails(text)

# -----------------------------
#  Main Analysis
# -----------------------------
def analyze_image(image_path):
    """Combine all detections and produce a unified result."""
    minor_detected = detect_minor(image_path)
    nsfw_detected = detect_nsfw(image_path)
    animal_detected = detect_animal(image_path)
    personal_details_detected = detect_personal_details(image_path)
    flagged_by_ai = minor_detected and nsfw_detected

    return {
        "filename": image_path,
        "minor_detected": minor_detected,
        "nsfw_detected": nsfw_detected,
        "animal_detected": animal_detected,
        "is_personal_details_detected": personal_details_detected,
        "flagged_by_ai": flagged_by_ai
    }

# -----------------------------
#  Worker Wrapper
# -----------------------------
def process_face_detection(image_path):
    """Wrapper expected by worker.py"""
    result = analyze_image(image_path)
    return {
        "minor_detected": result["minor_detected"],
        "is_nsfw": result["nsfw_detected"],
        "is_personal_details_detected": result["is_personal_details_detected"]
    }

# -----------------------------
#  CLI Testing
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='Path to image file')
    args = parser.parse_args()

    if args.image:
        results = analyze_image(args.image)
        print(results)


