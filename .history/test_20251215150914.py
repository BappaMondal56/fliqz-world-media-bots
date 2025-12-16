from animal_detect.animal_porn_detect import has_animal
from drugs_alcohol_smoking_detect.das_detector import is_das_detected
from face_detect.face_detect import predict_age_group

image_path = "test/images/sadie.jpg"
# result,_ = has_animal(image_path)
# print(f"Image has animal content: {result}") // gives true or false

# result= is_das_detected(image_path)
# print(f"Image has drugs/alcohol/smoking content: {result}") // gives 0 or 1

# result = predict_age_group(image_path)
# print(f"Predicted Minor: {result}") // gives true or false



