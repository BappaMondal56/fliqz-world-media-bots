import os
import cv2
import redis
import time
import json
from pathlib import Path

from animal_detect.animal_detector import is_animal_detected
from drugs_alcohol_smoking_detect.das_detector import is_das_detected
from face_detect.minor_detect import is_minor
from meetup_detect.personal_details_detect import detect_personal_info  
from nsfw.nsfw_detector_owlvit import is_nsfw_detected
from violance_detect.violation_detect import is_violence_detected
from weapon_detect.weapon_detector import is_weapon_detected

from dynamic_update import dynamic_update
from config import REDIS_HOST, REDIS_PORT, REDIS_DB, INPUT_QUEUE, REDIS_BRPOP_TIMEOUT

# -----------------------------
# Redis
# -----------------------------
r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    decode_responses=True
)


# -----------------------------
# ORIGINAL PATH HANDLING (RESTORED)
# -----------------------------
POSSIBLE_BASE_PATHS = [
    "/var/www/html/admin.fliqzworld.com/public/storage",
    "/var/www/html/admin.fliqzworld.com/storage",
    "/var/www/html/admin.fliqzworld.com/public_html/storage",
    "D:/codex/bots/NSFW-DETECT-BOT/var/www/html/admin.fliqzworld.com/public/storage",
    "C:/CodeX/fliqz-world-media-bots"
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
def process_redis(payload: dict):

    # 1. Map table name
    payload["table_name"] = payload.get("table")

    # 2. Primary key mapping
    payload["primary_key"] = "id"
    payload["key_value"] = payload.get("id")

    # 3. Extract file path from nested data
    data = payload.get("data", {})
    payload["file_path"] = data.get("file")

    # Safety checks
    if not payload["table_name"] or not payload["key_value"]:
        print("❌ Missing DB identifiers, skipping")
        return

    if not payload["file_path"]:
        print("❌ No file in payload, skipping")
        return

    # -------------------------------------------------
    # FILE PATH NORMALIZATION
    # -------------------------------------------------
    original_file = payload["file_path"]
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
        animal_detected = is_animal_detected(file_path)
    except Exception as e:
        print("Animal error:", e)

    # 🍻 Drugs / Alcohol / Smoking
    try:
        das_detected = is_das_detected(file_path)
    except Exception as e:
        print("DAS error:", e)

    # 👶 Minor
    try:
        minor_detected = is_minor(file_path)
    except Exception as e:
        print("Minor error:", e)

    # 📄 Personal info (OCR)
    try:
        personal_info_detected = detect_personal_info(file_path)
    except Exception as e:
        print("PII error:", e)

    # 🔞 NSFW
    try:
        nsfw_detected = is_nsfw_detected(file_path)
    except Exception as e:
        print("NSFW error:", e)

    # ⚔️ Violence
    try:
        violence_detected = is_violence_detected(file_path)
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
    print("🚀 Media Moderation Worker started")
    print("📥 Listening on:", INPUT_QUEUE)

    while True:
        try:
            item = r.brpop(INPUT_QUEUE, timeout=REDIS_BRPOP_TIMEOUT)
            if not item:
                time.sleep(0.1)
                continue

            _, message = item

            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                print("⚠️ Invalid JSON")
                continue

            process_redis(payload)

        except Exception as e:
            print("❌ Worker error:", e)
            time.sleep(1)

# -----------------------------
# ENTRY
# -----------------------------
if __name__ == "__main__":
    worker()


#"{\"type\":\"attachment\",\"table\":\"attachments\",\"id\":\"a83edaf06cb94905ad6d9f20b9e7dfc9\",\"data\":
# {\"file\":\"uploads\\/posts\\/images\\/17665066961999.jpg\",\"type\":\"images\",\"post_id\":512,\"user_id\":6}}"