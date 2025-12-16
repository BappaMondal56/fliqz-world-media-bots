# # from animal_detect.animal_porn_detect import has_animal
# # from drugs_alcohol_smoking_detect.das_detector import is_das_detected
# from face_detect.face_detect import *
# from face_detect.minor_detect import is_minor
# # from meetup_detect.personal_details_detect import detect_personal_info  
from nsfw.nsfw_detector import is_nsfw_image
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

# minor_detected = is_minor("storage/posts/images/nsfw3.jpg")
# nsfw = detect_nsfw("storage/posts/images/nsfw3.jpg")
# result = predict_age_group("storage/posts/images/nsfw3.jpg")
# print("Predicted Minor:", result)
# print("NSFW detected:", nsfw)
# print("Minor detected:", minor_detected)


import torch
from PIL import Image
import cv2
import numpy as np
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from collections import defaultdict
import os

# Load Model
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16")

# Cleaned Labels
labels = [
    "Porn", "Nudity", "Sexy", "Breasts", "Buttocks", "Genitalia", "Sexual Activity"
]

# Class-Specific Thresholds
CLASS_THRESHOLDS = {
    # High confidence needed (easily confused)
    "Water": 0.60,
    "Lollypop": 0.55,
    
    # Alcohol - moderate confidence
    "Alcohol": 0.40,
    "Alcohol bottle": 0.45,
    "Beer": 0.38,
    "Wine": 0.45,
    "Liquor": 0.45,
    
    # Smoking - high confidence (very distinct)
    "cigarette": 0.50,
    "smoking": 0.50,
    "cigar": 0.50,
    "vape": 0.50,
    
    # Drugs - very high (rare/ambiguous)
    "brown sugar drugs": 0.65,
    "drug packet": 0.65,
    "weed joint": 0.50,
    "cannabis": 0.55,
    
    # Injections - moderate
    "injection": 0.45,
    "syringe": 0.48,
    "needle": 0.50,
    "Injection Needle": 0.48,
    
    # Pills
    "Tablet": 0.50,
    "Pill": 0.50,
    "Capsule": 0.40,
}


def calculate_iou(box1, box2):
    """Calculate Intersection over Union"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def remove_overlapping_detections(detections, iou_threshold=0.5):
    """Keep only highest confidence detection for overlapping boxes"""
    if not detections:
        return []
    
    # Sort by confidence descending
    sorted_dets = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    for det in sorted_dets:
        overlap = False
        for kept_det in keep:
            if calculate_iou(det['box'], kept_det['box']) > iou_threshold:
                overlap = True
                break
        if not overlap:
            keep.append(det)
    
    return keep


def filter_detections(detections):
    """Remove detections if negative labels have higher confidence"""
    negative_labels = {"Water", "Lollypop"}
    
    # Find max confidence for negative labels
    max_negative_conf = max(
        (d['confidence'] for d in detections if d['label'] in negative_labels),
        default=0.0
    )
    
    # If negative label is confident, filter aggressively
    if max_negative_conf > 0.50:
        threshold_boost = 0.15
    else:
        threshold_boost = 0.0
    
    filtered = []
    for det in detections:
        if det['label'] in negative_labels:
            continue
        base_threshold = CLASS_THRESHOLDS.get(det['label'], 0.44)
        if det['confidence'] >= (base_threshold + threshold_boost):
            filtered.append(det)
    
    return filtered


def aggregate_video_detections(results_video, min_frames=3, confidence_threshold=0.40):
    """
    Only report detections that appear consistently with reasonable confidence
    """
    # Group by label
    label_frames = defaultdict(list)
    for det in results_video:
        label_frames[det['label']].append(det['confidence'])
    
    # Filter based on consistency
    final_detections = {}
    for label, confidences in label_frames.items():
        if len(confidences) >= min_frames:
            avg_conf = sum(confidences) / len(confidences)
            max_conf = max(confidences)
            
            # Only accept if average is decent AND max is high
            if avg_conf >= confidence_threshold and max_conf >= 0.42:
                final_detections[label] = {
                    'frames_detected': len(confidences),
                    'avg_confidence': round(avg_conf, 3),
                    'max_confidence': round(max_conf, 3)
                }
    
    return final_detections


def detect_das_image(image_path):
    """Detect DAS in a single image"""
    image = Image.open(image_path).convert("RGB")

    inputs = processor(text=labels, images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.25
    )[0]

    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_name = labels[label]
        threshold = CLASS_THRESHOLDS.get(label_name, 0.44)
        
        if score >= threshold:
            box = [round(i, 2) for i in box.tolist()]
            detections.append({
                "label": label_name,
                "confidence": float(score),
                "box": box
            })

    # Remove overlapping detections
    detections = remove_overlapping_detections(detections)
    
    # Filter based on negative labels
    detections = filter_detections(detections)
    
    return detections


def detect_das_video(video_path, skip_frames=20):
    """
    Detect DAS in video with improved filtering
    """
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    results_video = []

    print(f"Processing video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % skip_frames == 0:
            print(f"Processing frame {frame_id}/{total_frames}...", end='\r')
            
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = processor(text=labels, images=pil_img, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)

            target_sizes = torch.tensor([pil_img.size[::-1]])
            results = processor.post_process_object_detection(
                outputs, threshold=0.25, target_sizes=target_sizes
            )[0]

            frame_detections = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                label_name = labels[label]
                threshold = CLASS_THRESHOLDS.get(label_name, 0.44)
                
                if score >= threshold:
                    frame_detections.append({
                        "frame": frame_id,
                        "label": label_name,
                        "confidence": float(score),
                        "box": [float(i) for i in box.tolist()]
                    })
            
            # Remove overlapping detections per frame
            frame_detections = remove_overlapping_detections(frame_detections)
            
            # Filter based on negative labels
            frame_detections = filter_detections(frame_detections)
            
            results_video.extend(frame_detections)

        frame_id += 1

    cap.release()
    print(f"\nProcessing complete!")
    
    # Aggregate across video
    aggregated = aggregate_video_detections(results_video, min_frames=3, confidence_threshold=0.38)
    
    return results_video, aggregated


def print_video_summary(raw_detections, aggregated_detections):
    """Print summary of video analysis"""
    print("\n" + "="*60)
    print("VIDEO ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\nTotal raw detections: {len(raw_detections)}")
    
    if aggregated_detections:
        print("\n✓ DETECTED SUBSTANCES (after filtering):")
        for label, stats in aggregated_detections.items():
            print(f"\n  • {label}")
            print(f"    - Frames detected: {stats['frames_detected']}")
            print(f"    - Avg confidence: {stats['avg_confidence']:.3f}")
            print(f"    - Max confidence: {stats['max_confidence']:.3f}")
    else:
        print("\n✗ No substances detected after filtering")
    
    print("\n" + "="*60)


def is_das_detected(path):
    """
    Main detection function that handles both images and videos
    
    Args:
        path (str): Path to image or video file
        
    Returns:
        int: 1 if substances detected, 0 if not detected
    """
    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        return False
    
    # Get file extension
    ext = os.path.splitext(path)[1].lower()
    
    # Image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    # Video extensions
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    if ext in image_extensions:
        # Image Detection
        print("\n" + "="*60)
        print("IMAGE DETECTION")
        print("="*60)
        
        detections = detect_das_image(path)
        print(f"\nDetections found: {len(detections)}")
        for det in detections:
            print(f"  • {det['label']}: {det['confidence']:.3f}")
        
        return True if detections else False
        
    elif ext in video_extensions:
        # Video Detection
        print("\n" + "="*60)
        print("VIDEO DETECTION")
        print("="*60)
        
        raw_detections, aggregated = detect_das_video(
            video_path=path,
            skip_frames=20
        )
        
        print_video_summary(raw_detections, aggregated)
        
        return True if aggregated else False
        
    else:
        print(f"Error: Unsupported file format: {ext}")
        print(f"Supported formats: {image_extensions + video_extensions}")
        return False


result = is_das_detected("storage/posts/images/nsfw5.jpg")
nsfw = is_nsfw_image("storage/posts/images/nsfw5.jpg")
print("nsfw detected:", nsfw)