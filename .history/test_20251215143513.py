from animal_detect.animal_porn_detect import has_animal

image_path = "test/images/dog.jpg"
result,_ = has_animal(image_path)
print(f"Image has animal content: {result}")