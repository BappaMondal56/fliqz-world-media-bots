from animal_detect.animal_porn_detect import has_animal
from drugs_alcohol_smoking_detect.das_detector import is_das_detected

image_path = "test/images/dog.jpg"
# result,_ = has_animal(image_path)
# print(f"Image has animal content: {result}")
result,_ = is_das_detected(image_path)
print(f"Image has drugs/alcohol/smoking content: {result}")
