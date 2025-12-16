import os
import cv2
import redis
import time
import json
from pathlib import Path

from animal_detect.animal_porn_detect import has_animal
from drugs_alcohol_smoking_detect.das_detector import is_das_detected
from face_detect.face_detect import predict_age_group
from meetup_detect.personal_details_detect import detect_personal_info  
from nsfw.nsfw_detector import is_nsfw_image
from violance_detect.violation_detect import is_violence_detected
from weapon_detect.weapon_detector import is_weapon_detected

from dynamic_update import dynamic_update
from config import *

# -----------------------------
# Redis
# -----------------------------
r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    password=REDIS_PASSWORD,
    decode_responses=True
)

INPUT_QUEUE = "fliqz_moderation_image_video_queue"

# -----------------------------
# ORIGINAL PATH HANDLING (RESTORED)
# -----------------------------
POSSIBLE_BASE_PATHS = [
    "/var/www/html/admin.fliqzworld.com/public/storage",
    "/var/www/html/admin.fliqzworld.com/storage",
    "/var/www/html/admin.fliqzworld.com/public_html/storage",
    "D:/codex/bots/NSFW-DETECT-BOT/var/www/html/admin.fliqzworld.com/public/storage"
]

def get_valid_base_path():
    """Auto-detect which base path exists."""
    print("🔍 Checking possible base paths...")
    for base in POSSIBLE_BASE_PATHS:
        if os.path.exists(base):
            print(f"✅ Using detected base path: {base}")
            return base
        else:
            print(f"❌ Not found: {base}")
    print("⚠️ No valid storage path found! Using default first one.")
    return POSSIBLE_BASE_PATHS[0]

SERVER_STORAGE_PATH = get_valid_base_path()

def normalize_file_path(original_file: str) -> str:
    """Convert relative upload paths to absolute filesystem paths."""
    if not original_file:
        return ""

    clean_path = (
        original_file.replace("\\\\", "/")
        .replace("\\", "/")
        .replace("\\/", "/")
        .replace("//", "/")
        .strip()
        .lstrip("/")
    )

    print(f"🧭 Normalizing file path: {original_file} → {clean_path}")

    for base in POSSIBLE_BASE_PATHS:
        full_path = os.path.join(base, clean_path).replace("\\", "/")
        if os.path.exists(full_path):
            print(f"✅ Matched existing file path: {full_path}")
            return full_path

    fallback_path = os.path.join(SERVER_STORAGE_PATH, clean_path).replace("\\", "/")
    print(f"⚠️ Fallback path used: {fallback_path}")
    return fallback_path

# =====================================================
# PROCESS ONE REDIS MESSAGE
# =====================================================
def process_redis_message(payload: dict):

    data = payload.get("data", {})

    # DB identifiers (required)
    if not all(k in payload for k in ["tabel_name", "primary_key", "key_value"]):
        print("❌ Missing DB identifiers, skipping")
        return

    original_file = data.get("file")
    if not original_file:
        print("❌ No file in payload, skipping")
        return

    file_path = normalize_file_path(original_file)
    print(f"\n🖼️ Processing file: {file_path}")

    if not os.path.exists(file_path):
        print("❌ File not found after normalization")
        return

    # -----------------------------
    # FLAGS (DEFAULT FALSE)
    # -----------------------------
    animal_detected = False
    das_detected = False
    minor_detected = False
    personal_info_detected = False
    nsfw_detected = False
    violence_detected = False
    weapon_detected = False

    # 🐾 Animal
    try:
        animal_detected, _ = has_animal(file_path)
    except Exception as e:
        print("Animal error:", e)

    # 🍻 Drugs / Alcohol / Smoking
    try:
        das_detected = is_das_detected(file_path)
    except Exception as e:
        print("DAS error:", e)

    # 👶 Minor
    try:
        face_result = process_face_detection(file_path)
        minor_detected = face_result.get("minor_detected", False)
    except Exception as e:
        print("Minor error:", e)

    # 📄 Personal info (OCR)
    try:
        text = extract_text_from_file(file_path)
        personal_info_detected = isPersonalDetails(text)
    except Exception as e:
        print("PII error:", e)

    # 🔞 NSFW
    try:
        nsfw_detected = is_nsfw_image(file_path)
    except Exception as e:
        print("NSFW error:", e)

    # ⚔️ Violence
    try:
        label, _ = predict_violation(file_path)
        violence_detected = (label == "Violence")
    except Exception as e:
        print("Violence error:", e)

    # 🔫 Weapon
    try:
        weapon_detected = is_weapon_detected(file_path)
    except Exception as e:
        print("Weapon error:", e)

    # -----------------------------
    # DB UPDATE (UPDATE-ONLY)
    # -----------------------------
    success, status = dynamic_update(
        payload=payload,
        animal_detected=animal_detected,
        das_detected=das_detected,
        minor_detected=minor_detected,
        personal_info_detected=personal_info_detected,
        nsfw_detected=nsfw_detected,
        violence_detected=violence_detected,
        weapon_detected=weapon_detected
    )

    print("💾 DB Update:", status if success else f"FAILED ({status})")

# =====================================================
# WORKER LOOP
# =====================================================
def worker():
    print("🚀 Image Moderation Worker started")
    print("📥 Listening on:", INPUT_QUEUE)

    while True:
        try:
            item = r.brpop(INPUT_QUEUE, timeout=5)
            if not item:
                time.sleep(0.1)
                continue

            _, message = item

            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                print("⚠️ Invalid JSON")
                continue

            process_redis_message(payload)

        except Exception as e:
            print("❌ Worker error:", e)
            time.sleep(1)

# -----------------------------
# ENTRY
# -----------------------------
if __name__ == "__main__":
    worker()
