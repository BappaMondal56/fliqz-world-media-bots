from animal_detect.animal_porn_detect import has_animal

image_path = "test/images/ak47mulla.png"
result = has_animal(image_path)
print(f"Image {image_path} has animal content: {result}")