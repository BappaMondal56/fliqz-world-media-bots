from animal_detect.animal_porn_detect import has_animal
from drugs_alcohol_smoking_detect.das_detector import is_das_detected
from face_detect.face_detect import predict_age_group
from meetup_detect.personal_details_detect import detect_personal_info  
from nsfw.nsfw_detector import predict

image_path = "test/images/banana.jpg"
# result,_ = has_animal(image_path)
# print(f"Image has animal content: {result}") // gives true or false

# result= is_das_detected(image_path)
# print(f"Image has drugs/alcohol/smoking content: {result}") // gives 0 or 1

# result = predict_age_group(image_path)
# print(f"Predicted Minor: {result}") // gives true or false


# result = detect_personal_info(image_path)
# print(f"Image has personal details: {result}")  # gives true or false
