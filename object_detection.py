import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image, ImageOps
import shutil
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
# Constants
IMAGE_SIZE = (373, 454)  # Desired image size
DATASET_DIR = ''  # Base dataset directory

def filter_images_by_size(images_dir, desired_size=(373, 454)):
    filtered_images = []
    for img_filename in sorted(os.listdir(images_dir)):
        img_path = os.path.join(images_dir, img_filename)
        with Image.open(img_path) as img:
            if img.size == desired_size:
                filtered_images.append(img_path)
    return filtered_images[:612]  # Limit to the first 612 images

def copy_images(image_paths, target_dir, annotations_dir):
    images_target_dir = os.path.join(target_dir, 'images')
    annotations_target_dir = os.path.join(target_dir, 'labels')

    os.makedirs(images_target_dir, exist_ok=True)
    os.makedirs(annotations_target_dir, exist_ok=True)

    for path in image_paths:
        img_name = os.path.basename(path)
        img_target_path = os.path.join(images_target_dir, img_name)
        shutil.copy(path, img_target_path)

        # Copy corresponding annotation file
        annotation_name = os.path.splitext(img_name)[0] + '.txt'
        annotation_source_path = os.path.join(annotations_dir, annotation_name)
        annotation_target_path = os.path.join(annotations_target_dir, annotation_name)
        shutil.copy(annotation_source_path, annotation_target_path)


# Filtering images by size and preparing the dataset
images_dir = os.path.join(DATASET_DIR, 'images', 'Fractured')
filtered_images = filter_images_by_size(images_dir)

# Splitting the dataset
train_val_imgs, test_imgs = train_test_split(filtered_images, test_size=0.15, random_state=42)
train_imgs, val_imgs = train_test_split(train_val_imgs, test_size=0.176, random_state=42)  # Approximately 15% of 0.85

# Copying images and annotations to train, val, and test directories
copy_images(train_imgs, os.path.join(DATASET_DIR, 'train'), os.path.join(DATASET_DIR, 'Annotations', 'YOLO'))
copy_images(val_imgs, os.path.join(DATASET_DIR, 'val'), os.path.join(DATASET_DIR, 'Annotations', 'YOLO'))
copy_images(test_imgs, os.path.join(DATASET_DIR, 'test'), os.path.join(DATASET_DIR, 'Annotations', 'YOLO'))

class FracAtlasDataset(Dataset):
    def __init__(self, img_dir, annotation_dir, transform=None):
        self.img_dir = os.path.join(img_dir, 'images')  # Adjust to point to the images subdirectory
        self.annotation_dir = os.path.join(annotation_dir, 'labels')  # Adjust to point to the annotations subdirectory
        self.transform = transform
        self.imgs = list(sorted(os.listdir(self.img_dir)))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img = self.resize_and_pad(img, original_size=IMAGE_SIZE, desired_size=640)

        # Load annotations from the corresponding annotation file
        annotation_name = os.path.splitext(img_name)[0] + '.txt'
        annotation_path = os.path.join(self.annotation_dir, annotation_name)
        boxes = self.load_annotations_from_file(annotation_path)

        # Convert boxes to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        target = {'boxes': boxes, 'labels': torch.ones((len(boxes),), dtype=torch.int64)}

        if self.transform:
            img = self.transform(img)

        return img, target

    def resize_and_pad(self, img, original_size=(373, 454), desired_size=640):
        ratio = min(desired_size / original_size[0], desired_size / original_size[1])
        new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
        img = img.resize(new_size, Image.BILINEAR)
        delta_width = desired_size - new_size[0]
        delta_height = desired_size - new_size[1]
        padding = (delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))
        img = ImageOps.expand(img, padding)
        return img

    def load_annotations_from_file(self, annotation_path):
        with open(annotation_path, 'r') as file:
            # Parse annotation file and return bounding boxes
            # Example: Each line in the annotation file represents one bounding box
            # Format: class_index x_center y_center width height
            boxes = []
            for line in file:
                values = line.strip().split()
                class_index = int(values[0])
                x_center = float(values[1])
                y_center = float(values[2])
                width = float(values[3])
                height = float(values[4])
                boxes.append([x_center, y_center, width, height])
            return boxes


def transform(img):
    img = F.to_tensor(img)
    img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return img

def collate_fn(batch):
    imgs, targets = list(zip(*batch))
    imgs = torch.stack(imgs)
    return imgs, targets

# Define paths to the directories containing images and annotations for training and validation datasets
train_img_dir = os.path.join(DATASET_DIR, 'train')
train_annotation_dir = os.path.join(DATASET_DIR, 'train')

val_img_dir = os.path.join(DATASET_DIR, 'val')
val_annotation_dir = os.path.join(DATASET_DIR, 'val')

# Instantiate the training dataset with both image and annotation directories
train_dataset = FracAtlasDataset(img_dir=train_img_dir, annotation_dir=train_annotation_dir, transform=transform)

# Instantiate the validation dataset with both image and annotation directories
val_dataset = FracAtlasDataset(img_dir=val_img_dir, annotation_dir=val_annotation_dir, transform=transform)

# Use DataLoader to load data in batches
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)


# Example of iterating over the DataLoader
for imgs, targets in train_loader:
    print("Batch of images:", imgs.shape)  # Shape: [batch_size, 3, 640, 640]
    print("Targets:", targets)  # List of dictionaries with 'boxes' and 'labels'
    break  # Just to show one batch

# Initialize the YOLOv8s model; specify the path to the pre-trained model or configuration file
model = YOLO('yolov8s.pt')
# Train the model
results = model.train(data='', epochs=20)  # Specify the actual path to your YAML file and desired number of epochs

save_dir = ""
torch.save(model.model.state_dict(), os.path.join(save_dir, 'object_detection_model.pt'))