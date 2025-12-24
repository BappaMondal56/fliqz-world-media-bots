# from animal_detect.animal_porn_detect import has_animal
# from drugs_alcohol_smoking_detect.das_detector import is_das_detected
# # from face_detect.face_detect import *
# from face_detect.minor_detect import is_minor
# from meetup_detect.personal_details_detect import detect_personal_info  
# from nsfw.nsfw_detector_owlvit import is_nsfw_detected
# # # from violance_detect.violation_detect import is_violence_detected
# # # from weapon_detect.weapon_detector import is_weapon_detected
# # image_path = "test/images/AR1.jpg"
# # # result,_ = has_animal(image_path)
# # # print(f"Image has animal content: {result}") // gives true or false

# # # result= is_das_detected(image_path)
# # # print(f"Image has drugs/alcohol/smoking content: {result}") // gives 0 or 1

# # # result = predict_age_group(image_path)
# # # print(f"Predicted Minor: {result}") // gives true or false


# # # result = detect_personal_info(image_path)
# # # print(f"Image has personal details: {result}")  # gives true or false

# # # flag = is_nsfw_image("test/images/nsfw3.jpg")
# # # print(flag)   # True or False

# # # result = is_violence_detected(image_path)
# # # print("Violence detected:", result) # gives true or false

# # # result = is_weapon_detected(image_path)
# # # print("Weapon detected:", result) # gives True or False

# # minor_detected = is_minor("storage/posts/images/test3.jpeg")
# # nsfw = detect_nsfw("storage/posts/images/nsfw3.jpg")
# # result = predict_age_group("storage/posts/images/nsfw3.jpg")
# # print("Predicted Minor:", result)
# # print("NSFW detected:", nsfw)
# # print("Minor detected:", minor_detected)


# # result = is_das_detected("storage/posts/images/nsfw5.jpg")
# # nsfw = is_nsfw_detected("storage/posts/videos/v_test.mp4")
# # print("nsfw detected:", nsfw)

# # animal_detected,_ = has_animal("storage/posts/images/dog2.jpg")
# # print("animal detected:", animal_detected)

# # detect_personal_info = detect_personal_info("storage/posts/videos/v_test.mp4")
# # print("Personal info detected:", detect_personal_info)


# das_detected = is_das_detected(image)
# print("Drugs/Alcohol/Smoking detected:", das_detected)

import torch
from PIL import Image
import cv2
import numpy as np
from transformers import Owlv2Processor, Owlv2ForObjectDetection

# Load model
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16")

labels = [
    "Water", "Lollypop",
    "Alcohol", "Alcohol bottle", "Liquor", "Wine", "Beer",
    "cigarette", "smoking", "cigar", "vape",
    "weed joint", "cannabis", "drug packet", "brown sugar drugs",
    "syringe", "injection", "needle",
    "Tablet", "Pill", "Capsule"
]


def detect_and_draw_confidence_only(image_path, output_path="detected_conf.jpg"):
    # Load image
    image = Image.open(image_path).convert("RGB")
    img = np.array(image)

    # Prepare input
    inputs = processor(
        text=labels,
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])

    # 🚨 NO threshold → everything shown
    results = processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=0.0
    )[0]

    for score, box in zip(results["scores"], results["boxes"]):
        confidence = float(score)
        x1, y1, x2, y2 = map(int, box.tolist())

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw ONLY confidence
        cv2.putText(
            img,
            f"{confidence:.2f}",
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Save output
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"✅ Saved image with confidence-only detections: {output_path}")

    # Show
    cv2.imshow("Detections (Confidence Only)", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


detect_and_draw_confidence_only("storage/posts/images/test3.jpeg")
