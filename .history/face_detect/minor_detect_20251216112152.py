# Minor Detection using OpenCV DNN (Age Buckets)
# Visualisation & gender removed

import cv2
import argparse
import os
import tempfile
# -----------------------------
# Face detection (unchanged logic)
# -----------------------------
def detect_faces(net, frame, conf_threshold=0.7):
    frameHeight, frameWidth = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300),
        [104, 117, 123], True, False
    )

    net.setInput(blob)
    detections = net.forward()

    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])

    return faceBoxes


# -----------------------------
# Models & constants
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

faceProto = os.path.join(BASE_DIR, "opencv_face_detector.pbtxt")
faceModel = os.path.join(BASE_DIR, "opencv_face_detector_uint8.pb")
ageProto  = os.path.join(BASE_DIR, "age_deploy.prototxt")
ageModel  = os.path.join(BASE_DIR, "age_net.caffemodel")

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_BUCKETS = [
    '(0-2)', '(4-6)', '(8-12)', '(15-22)',
    '(23-32)', '(38-43)', '(48-58)', '(60-80)'
]

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet  = cv2.dnn.readNet(ageModel, ageProto)


def normalize_to_jpg(image_path):
    """
    Convert any image to a normalized JPG for consistent inference.
    Returns path to temporary JPG.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    temp_path = temp_file.name
    temp_file.close()

    # Save as high-quality JPG (important)
    cv2.imwrite(temp_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return temp_path


# -----------------------------
# Minor detection function
# -----------------------------
def is_minor(image_path):
    """
    Returns:
        True  -> Minor detected
        False -> Adult or no face
    """

    if not os.path.exists(image_path):
        return False
    
    # 🔒 Normalize image format (PNG/JPG → JPG)
    normalized_path = normalize_to_jpg(image_path)
    if normalized_path is None:
        return False

    frame = cv2.imread(normalized_path)
    if frame is None:
        return False

    faceBoxes = detect_faces(faceNet, frame)

    if not faceBoxes:
        print("No face detected")
        return False

    padding = 20
    h, w = frame.shape[:2]

    for box in faceBoxes:
        x1, y1, x2, y2 = box

        face = frame[
            max(0, y1 - padding):min(y2 + padding, h - 1),
            max(0, x1 - padding):min(x2 + padding, w - 1)
        ]

        if face.size == 0:
            continue

        faceBlob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227),
            MODEL_MEAN_VALUES,
            swapRB=False
        )

        ageNet.setInput(faceBlob)
        agePreds = ageNet.forward()
        ageBucket = AGE_BUCKETS[agePreds[0].argmax()]

        print(f"Detected age bucket: {ageBucket}")

        # 🔒 Policy-safe minor decision
        if ageBucket in ['(0-2)', '(4-6)', '(8-12)', '(15-22)']:
            return True

    return False


# # -----------------------------
# # CLI test
# # -----------------------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--image", required=True)
#     args = parser.parse_args()

#     result = is_minor(args.image)
#     print("Minor detected:", result)
