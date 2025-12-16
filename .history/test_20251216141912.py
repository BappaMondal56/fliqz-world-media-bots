from animal_detect.animal_porn_detect import has_animal
# # from drugs_alcohol_smoking_detect.das_detector import is_das_detected
# from face_detect.face_detect import *
from face_detect.minor_detect import is_minor
# # from meetup_detect.personal_details_detect import detect_personal_info  
from nsfw.nsfw_detector_owlvit import is_nsfw_detected
# # from violance_detect.violation_detect import is_violence_detected
# # from weapon_detect.weapon_detector import is_weapon_detected
# image_path = "test/images/AR1.jpg"
# # result,_ = has_animal(image_path)
# # print(f"Image has animal content: {result}") // gives true or false

# # result= is_das_detected(image_path)
# # print(f"Image has drugs/alcohol/smoking content: {result}") // gives 0 or 1

# # result = predict_age_group(image_path)
# # print(f"Predicted Minor: {result}") // gives true or false


# # result = detect_personal_info(image_path)
# # print(f"Image has personal details: {result}")  # gives true or false

# # flag = is_nsfw_image("test/images/nsfw3.jpg")
# # print(flag)   # True or False

# # result = is_violence_detected(image_path)
# # print("Violence detected:", result) # gives true or false

# # result = is_weapon_detected(image_path)
# # print("Weapon detected:", result) # gives True or False

minor_detected = is_minor("storage/posts/images/kid1.jpg")
# nsfw = detect_nsfw("storage/posts/images/nsfw3.jpg")
# result = predict_age_group("storage/posts/images/nsfw3.jpg")
# print("Predicted Minor:", result)
# print("NSFW detected:", nsfw)
print("Minor detected:", minor_detected)


# result = is_das_detected("storage/posts/images/nsfw5.jpg")
# nsfw = is_nsfw_detected("storage/posts/videos/Output-Test-Video.mp4")
# print("nsfw detected:", 

# animal_detected,_ = has_animal("storage/posts/videos/dog.mp4")
# print("animal detected:", animal_detected)