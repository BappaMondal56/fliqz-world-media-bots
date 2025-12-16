# config.py
import os

# MySQL / SQLAlchemy settings (use env or hardcode for local)
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_DATABASE = os.getenv("DB_DATABASE", "u212337367_myvault")

# Redis settings
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# LLaMA / local LLM endpoint
LLAMA_API_URL = os.getenv("LLAMA_API_URL", "http://localhost:11434/api/generate")

# Redis queue name (single worker)
TEXT_QUEUE = os.getenv("ATTACHMENTS_QUEUE", "fliqz_moderation_stream_image_queue")

# Tuneable: how long to block on BRPOP (seconds)
REDIS_BRPOP_TIMEOUT = int(os.getenv("REDIS_BRPOP_TIMEOUT", 5))