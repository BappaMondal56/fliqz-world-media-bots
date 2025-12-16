import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "age_detection_model.h5")

model = load_model(MODEL_PATH)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def has_human_face(image_path, min_size=(60, 60)):
    img = cv2.imread(image_path)
    if img is None:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=min_size
    )

    return len(faces) > 0


def is_minor(image_path, img_size=(128, 128)):
    """
    Returns:
        True  → Human minor detected (<18)
        False → Adult OR no face OR non-human
    """

    if not os.path.exists(image_path):
        return False

    # 🚫 No face → no age prediction
    if not has_human_face(image_path):
        return False

    img = load_img(image_path, target_size=img_size)
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    age_pred = float(model.predict(img, verbose=0)[0][0])

    # 🚫 Invalid predictions safety
    if age_pred < 0 or age_pred > 100:
        return False

    return age_pred < 18


minor_detected = is_minor("storage/")

print("Minor detected:", minor_detected)
