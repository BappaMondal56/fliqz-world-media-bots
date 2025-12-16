import cv2
import os
import numpy as np

# -----------------------------
# Load models
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

faceProto = os.path.join(BASE_DIR, "opencv_face_detector.pbtxt")
faceModel = os.path.join(BASE_DIR, "opencv_face_detector_uint8.pb")
ageProto  = os.path.join(BASE_DIR, "age_deploy.prototxt")
ageModel  = os.path.join(BASE_DIR, "age_net.caffemodel")

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-22)',
               '(23-32)', '(38-43)', '(48-58)', '(60-80)']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)


# -----------------------------
# Minor Detection
# -----------------------------
def is_minor(image_path, conf_threshold=0.7):
    """
    Returns:
        True  -> Minor detected
        False -> Adult or no face
    """

    if not os.path.exists(image_path):
        return False

    frame = cv2.imread(image_path)
    if frame is None:
        return False

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300), [104, 117, 123], swapRB=True, crop=False
    )

    faceNet.setInput(blob)
    detections = faceNet.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < conf_threshold:
            continue

        x1 = int(detections[0, 0, i, 3] * w)
        y1 = int(detections[0, 0, i, 4] * h)
        x2 = int(detections[0, 0, i, 5] * w)
        y2 = int(detections[0, 0, i, 6] * h)

        face = frame[max(0, y1):min(y2, h), max(0, x1):min(x2, w)]
        if face.size == 0:
            continue

        faceBlob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False
        )

        ageNet.setInput(faceBlob)
        agePreds = ageNet.forward()
        ageBucket = AGE_BUCKETS[agePreds[0].argmax()]

        print(f"Detected age bucket: {ageBucket}")

        # ✅ Minor decision
        if ageBucket in ['(0-2)', '(4-6)', '(8-12)']:
            return True

    return False
