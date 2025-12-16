# meetup_detect/personal_details_detect.py

import re
import spacy
import platform
import pytesseract
from PIL import Image
import requests
from io import BytesIO

# Load NLP model once
nlp = spacy.load("en_core_web_sm")

# Regex patterns
email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
url_pattern = re.compile(r'(https?:\/\/[^\s]+|www\.[^\s]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:\/[^\s]*)?)')
phone_pattern = re.compile(r'\+?\d[\d\s\-()]{7,}\d')

PLATFORM_DOMAIN = "myvault-web.codextechnolife.com"

number_words = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million", "billion"
}

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    # On Linux (Ubuntu)
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# -----------------------------
# 🔹 Helper Functions
# -----------------------------
def isEmail(text: str) -> bool:
    return bool(email_pattern.search(text))

def hasPhoneNumber(text: str) -> bool:
    return bool(phone_pattern.search(text))

def hasNumber(n) -> bool:
    if isinstance(n, int):
        return True
    if isinstance(n, str):
        if PLATFORM_DOMAIN in n:
            return False
        return any(ch.isdigit() for ch in n)
    return False

def hasNumberWords(text: str) -> bool:
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return any(word in number_words for word in words)

def hasForbiddenURL(text: str) -> bool:
    for match in url_pattern.finditer(text):
        url = match.group(0).rstrip('.,!?')
        if not re.search(r'\.[a-zA-Z]{2,}', url):
            continue
        if PLATFORM_DOMAIN not in url:
            return True
    return False

def hasAddress(text: str) -> bool:
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in {"GPE", "LOC", "FAC"}:
            return True
    return False


# -----------------------------
# 🔹 Main Functions
# -----------------------------
def isPersonalDetails(text: str) -> bool:
    """Check if text contains personal information."""
    return any([
        hasForbiddenURL(text),
        isEmail(text),
        hasPhoneNumber(text),
        hasNumberWords(text),
        hasNumber(text),
        hasAddress(text)
    ])


def extract_text_from_file(file_path_or_url: str) -> str:
    """Extract text from image using OCR (Tesseract)."""
    try:
        if file_path_or_url.startswith("http"):
            response = requests.get(file_path_or_url)
            img = Image.open(BytesIO(response.content))
        else:
            img = Image.open(file_path_or_url)
        return pytesseract.image_to_string(img)
    except Exception as e:
        print(f"Error in OCR: {e}")
        return ""
