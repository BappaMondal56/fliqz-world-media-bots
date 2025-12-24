from animal_detect.animal_detector import is_animal_detected
# from drugs_alcohol_smoking_detect.das_detector import is_das_detected
# # from face_detect.face_detect import *
from face_detect.minor_detect import is_minor
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

# minor_detected = is_minor("storage/posts/images/minor.jpeg")
# print("Minor detected:", minor_detected)
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
# import torch
# from PIL import Image
# from transformers import Owlv2Processor, Owlv2ForObjectDetection

# # Load model
# processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16")
# model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16")

# # Animal-related prompts
# ANIMAL_LABELS = [
#     "animal",
#     "dog",
#     "cat",
#     "cow",
#     "horse",
#     "goat",
#     "sheep",
#     "pig",
#     "monkey",
#     "bird",
#     "fish",
#     "elephant",
#     "tiger",
#     "lion",
#     "bear"
# ]


# def get_animal_confidence(image_path):
#     """
#     Returns max confidence for Animal presence in image.
#     Always returns a float (0.0 if nothing found)
#     """

#     image = Image.open(image_path).convert("RGB")

#     inputs = processor(
#         text=ANIMAL_LABELS,
#         images=image,
#         return_tensors="pt"
#     )

#     with torch.no_grad():
#         outputs = model(**inputs)

#     target_sizes = torch.tensor([image.size[::-1]])

#     results = processor.post_process_object_detection(
#         outputs,
#         target_sizes=target_sizes,
#         threshold=0.0  # allow everything
#     )[0]

#     animal_scores = [float(score) for score in results["scores"]]

#     # If no detections → 0.0
#     animal_confidence = max(animal_scores) if animal_scores else 0.0

#     return round(animal_confidence, 4)




# conf = get_animal_confidence("storage/posts/images/dog4.jpeg")
# print("Animal confidence:", conf)


animal_detected = is_animal_detected("storage/posts/images/goat.jpg")
print("Animal detected:", animal_detected)