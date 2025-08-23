"""
Convert BDD100K dataset annotations to YOLO format with COCO80 labels.

Usage:
    python3 5_bdd100k_to_yolo.py

Requirements:
    - BDD100K dataset should be located at '../datasets/bdd/100k'

Output:
    - Converts BDD100K to YOLO format in '../datasets/bdd100k_yolo'
    - Creates train/val/test splits with COCO80 label mapping
"""

import os
import json
import shutil
import yaml
from PIL import Image
from utils import COCO80_CLASSES

script_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(os.path.dirname(script_dir), 'datasets')
bdd100k_dir = os.path.join(datasets_dir, 'bdd', '100k')
output_dir = os.path.join(datasets_dir, 'bdd100k_yolo')

# BDD100K category to COCO80 mapping
BDD_TO_COCO80 = {
    'person': 0,
    'car': 2,
    'bus': 5,
    'truck': 7,
    'traffic light': 9,
    'train': 6,
    'rider': 0,  # -> person
    'bike': 1,  # -> bicycle
    'motor': 3,  # -> motorcycle
    'traffic sign': -1,  # cannot map to COCO80 stop sign
}

def create_output_dirs():
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

def convert_bbox_to_yolo(bbox, img_width, img_height):
    x1, y1, x2, y2 = bbox
    x_center = (x1 + x2) / 2.0 / img_width
    y_center = (y1 + y2) / 2.0 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return x_center, y_center, width, height

def process_bdd_annotations(data, img_width, img_height):
    yolo_annotations = []
    
    for frame in data.get('frames', []):
        for obj in frame.get('objects', []):
            category = obj.get('category', '')
            
            if category not in BDD_TO_COCO80 or BDD_TO_COCO80[category] == -1:
                continue
            
            bbox_data = obj.get('box2d')
            if not bbox_data:
                continue
            
            x1 = bbox_data['x1']
            y1 = bbox_data['y1'] 
            x2 = bbox_data['x2']
            y2 = bbox_data['y2']
            
            if x1 >= x2 or y1 >= y2:
                continue
            
            x_center, y_center, width, height = convert_bbox_to_yolo(
                [x1, y1, x2, y2], img_width, img_height
            )
            
            if x_center < 0 or x_center > 1 or y_center < 0 or y_center > 1:
                continue
            if width <= 0 or height <= 0 or width > 1 or height > 1:
                continue
            
            class_id = BDD_TO_COCO80[category]
            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            yolo_annotations.append(yolo_line)
    
    return yolo_annotations

def find_labels_directory(base_dir, split):
    datasets_dir = os.path.dirname(base_dir)
    possible_dirs = [
        os.path.join(base_dir, 'labels', split),
        os.path.join(datasets_dir, 'bdd', 'labels', '100k', split)
    ]
    
    for labels_dir in possible_dirs:
        if os.path.exists(labels_dir):
            return labels_dir
    return None

def process_split(split):
    print(f"Processing {split} split...")
    
    labels_dir = find_labels_directory(bdd100k_dir, split)
    if not labels_dir:
        print(f"Warning: Labels directory not found for {split}, skipping...")
        return
    
    images_dir = os.path.join(bdd100k_dir, 'images', split)
    if not os.path.exists(images_dir):
        print(f"Warning: Images directory not found for {split}: {images_dir}")
        return
    
    output_images_dir = os.path.join(output_dir, 'images', split)
    output_labels_dir = os.path.join(output_dir, 'labels', split)
    
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    image_names = {os.path.splitext(f)[0] for f in image_files}
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.json')]
    
    print(f"Found {len(image_files)} images and {len(label_files)} label files for {split}")
    
    processed = 0
    skipped = 0
    no_matching_image = 0
    no_valid_objects = 0
    
    for label_file in label_files:
        try:
            label_name = os.path.splitext(label_file)[0]
            
            if label_name not in image_names:
                no_matching_image += 1
                continue
                
            image_file = f"{label_name}.jpg"
            image_path = os.path.join(images_dir, image_file)
            label_path = os.path.join(labels_dir, label_file)
            
            with open(label_path, 'r') as f:
                data = json.load(f)
            
            with Image.open(image_path) as img:
                img_width, img_height = img.size
            
            yolo_annotations = process_bdd_annotations(data, img_width, img_height)
            
            output_label_path = os.path.join(output_labels_dir, f"{label_name}.txt")
            with open(output_label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
            
            output_image_path = os.path.join(output_images_dir, image_file)
            if not os.path.exists(output_image_path):
                shutil.copy2(image_path, output_image_path)
            
            if len(yolo_annotations) == 0:
                no_valid_objects += 1
            
            processed += 1
            
            if processed % 2000 == 0:
                print(f"Processed {processed} files for {split}")
                
        except Exception as e:
            print(f"Error processing {label_file}: {e}")
            skipped += 1
            continue
    
    # Handle images without labels for BDD100K
    label_names = {os.path.splitext(f)[0] for f in label_files}
    images_without_labels = image_names - label_names
    
    for image_name in images_without_labels:
        output_label_path = os.path.join(output_labels_dir, f"{image_name}.txt")
        with open(output_label_path, 'w') as f:
            f.write('')  # Empty file
        
        image_file = f"{image_name}.jpg"
        image_path = os.path.join(images_dir, image_file)
        output_image_path = os.path.join(output_images_dir, image_file)
        if not os.path.exists(output_image_path):
            shutil.copy2(image_path, output_image_path)
    
    no_valid_objects += len(images_without_labels)
    
    print(f"Completed {split}: {processed} processed, {skipped} skipped, "
          f"{no_matching_image} no matching image, {no_valid_objects} no valid objects")

def create_yaml_config():
    config = {
        'path': output_dir,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 80,
        'names': COCO80_CLASSES
    }
    
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created dataset config: {yaml_path}")

def main():
    print("Converting BDD100K to YOLO format with COCO80 labels...")
    
    if not os.path.exists(bdd100k_dir):
        print(f"Error: BDD100K directory not found: {bdd100k_dir}")
        return
    
    create_output_dirs()
    
    for split in ['train', 'val', 'test']:
        labels_dir = find_labels_directory(bdd100k_dir, split)
        if labels_dir:
            process_split(split)
        else:
            print(f"Warning: {split} split not found, skipping...")
    
    create_yaml_config()
    print(f"Conversion complete! Dataset saved to: {output_dir}")

if __name__ == "__main__":
    main()