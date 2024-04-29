import os
import cv2
import numpy as np
import json

# Constants
IMAGE_SIZE = (373, 454)  # Corrected image size to match model input shape
DATASET_DIR = ''

# Load COCO annotations
def load_annotations():
    annotations_file = os.path.join(DATASET_DIR, 'Annotations', 'COCO JSON', 'COCO_fracture_masks.json')
    with open(annotations_file, 'r') as file:
        data = json.load(file)
        # Create a dictionary to map image IDs to annotations
        annotations_dict = {}
        # Create a dictionary to map image names to image IDs
        image_id_to_name = {image['id']: image['file_name'] for image in data['images']}
        for annotation in data['annotations']:
            image_id = annotation['image_id']
            image_name = image_id_to_name.get(image_id)
            if image_name is None:
                continue  # Skip annotations for images not found in the images list
            bbox = annotation['bbox']
            if image_name not in annotations_dict:
                annotations_dict[image_name] = []
            annotations_dict[image_name].append(bbox)
        return annotations_dict




# Draw bounding boxes on images
def draw_boxes(image_path, boxes):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image", image_path)
        return
    for box in boxes:
        x, y, w, h = map(int, box)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('Image with Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_name = 'IMG0002036.jpg'  # Replace with the desired image name
image_path = os.path.join(DATASET_DIR, 'images', 'Fractured', image_name)
annotations_dict = load_annotations()


print("Image Path:", image_path)  # Check if the image path is correct

# Load the image
image = cv2.imread(image_path)
if image is None:
    print("Error: Unable to load image", image_path)
else:
    print("Image loaded successfully")

# Draw bounding boxes on the image if it's loaded successfully
if image is not None:
    boxes = annotations_dict.get(image_name, [])
    if boxes:
        draw_boxes(image_path, boxes)
    else:
        print("No annotations found for image", image_name)


