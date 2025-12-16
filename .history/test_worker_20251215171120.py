import os
import cv2
import json
import time
import redis
from pathlib import Path

from nsfw.nsfw_detector import NSFWDetector
from animal_detect.animal_porn_detect import has_animal
from face_detect.face_detect import process_face_detection
from violance_detect.violation_detect import predict_violation
from meetup_detect.personal_details_detect import extract_text_from_file, isPersonalDetails
from weapon_detect.weapon_detector import is_weapon_detected
from drugs_alcohol_smoking_detect.das_detector import is_das_detected
from db.save_to_mysql import update_ai_detection
from config import *

# -----------------------------
# Redis
# -----------------------------


# -----------------------------
# Storage paths
# -----------------------------
POSSIBLE_BASE_PATHS = [
    "/var/www/html/admin.fliqzworld.com/public/storage",
    "/var/www/html/admin.fliqzworld.com/storage",
    "/var/www/html/admin.fliqzworld.com/public_html/storage",
    "D:/codex/bots/NSFW-DETECT-BOT/var/www/html/admin.fliqzworld.com/public/storage"
]

BASE_STORAGE_PATH = next(
    (p for p in POSSIBLE_BASE_PATHS if os.path.exists(p)),
    POSSIBLE_BASE_PATHS[0]
)

# =====================================================
# 1️⃣ Process ONE Redis message
# =====================================================
def process_redis_message(payload: dict):

    data = payload.get("data", {})
    redis_id = data.get("id")
    table_name = data.get("table", "attachments")
    original_file = data.get("file")

    if not redis_id or not original_file:
        print("❌ Missing id or file in payload, skipping")
        return

    # Normalize path
    clean_path = (
        original_file.replace("\\", "/")
        .replace("//", "/")
        .lstrip("/")
    )

    file_path = os.path.join(BASE_STORAGE_PATH, clean_path)

    print(f"\n🖼️ Processing image: {file_path}")

    if not os.path.exists(file_path):
        print(f"❌ FILE NOT FOUND: {file_path}")
        return

    # -----------------------------
    # Initialize flags
    # -----------------------------
    animal_detected = False
    nsfw_detected = False
    minor_detected = False
    violence_detected = False
    personal_info_detected = False
    weapon_detected = False
    das_detected = False

    # 🐾 Animal
    try:
        animal_detected, _ = has_animal(file_path)
    except Exception as e:
        print("❌ Animal error:", e)

    # 👶 NSFW + Minor
    try:
        face_result = process_face_detection(file_path)
        nsfw_detected = face_result.get("is_nsfw", False)
        minor_detected = face_result.get("minor_detected", False)
    except Exception as e:
        print("❌ Face error:", e)

    # ⚔️ Violence
    try:
        label, _ = predict_violation(file_path)
        violence_detected = (label == "Violence")
    except Exception as e:
        print("❌ Violence error:", e)

    # 📄 Personal info (OCR)
    try:
        text = extract_text_from_file(file_path)
        personal_info_detected = isPersonalDetails(text)
    except Exception as e:
        print("❌ PII error:", e)

    # 🔫 Weapon
    try:
        weapon_detected = is_weapon_detected({"file": file_path})
    except Exception as e:
        print("❌ Weapon error:", e)

    # 🍻 DAS
    try:
        das_detected = is_das_detected({"file": file_path})
    except Exception as e:
        print("❌ DAS error:", e)

    # -----------------------------
    # Build update payload
    # -----------------------------
    update_json = {
        "id": redis_id,
        "animal_detected": int(animal_detected),
        "nsfw_detected": int(nsfw_detected),
        "minor_detected": int(minor_detected),
        "violance_detected": int(violence_detected),
        "is_personal_details_detected": int(personal_info_detected),
        "is_weapon_detected": int(weapon_detected),
        "is_das_detected": int(das_detected),
        "flagged_by_ai": int(
            animal_detected or nsfw_detected or minor_detected
            or violence_detected or personal_info_detected
            or weapon_detected or das_detected
        )
    }

    # Push to output queue
    r.lpush(OUTPUT_QUEUE, json.dumps(update_json))
    print("📤 Pushed result to Redis")

    # Update DB (UPDATE-ONLY)
    success, status = update_ai_detection(table_name, update_json, payload)
    print("💾 DB update:", status if success else f"FAILED ({status})")


# =====================================================
# 2️⃣ Worker loop
# =====================================================
def worker_loop():
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
                print("⚠️ Invalid JSON:", message)
                continue

            process_redis_message(payload)

        except Exception as e:
            print("❌ Worker error:", e)
            time.sleep(1)


# -----------------------------
# Entry
# -----------------------------
if __name__ == "__main__":
    worker_loop()
