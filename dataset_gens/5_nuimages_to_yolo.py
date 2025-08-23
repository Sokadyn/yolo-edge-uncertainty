"""
Convert nuImages dataset annotations to YOLO format with COCO80 labels.

Usage:
    python3 5_nuimages_to_yolo.py

Requirements:
    - nuImages dataset should be located at '../datasets/nuimages'

Output:
    - Converts nuImages to YOLO format in '../datasets/nuimages_yolo'
    - Creates train/val/test splits with COCO80 label mapping
"""

import os
import json
import shutil
from PIL import Image
from utils import COCO80_CLASSES, convert_bbox_to_yolo, create_yaml_config

script_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(os.path.dirname(script_dir), 'datasets')
nuimages_dir = os.path.join(datasets_dir, 'nuimages')
output_dir = os.path.join(datasets_dir, 'nuimages_yolo')

NUIMAGES_TO_COCO80 = {
    'human.pedestrian.adult': 0,
    'human.pedestrian.child': 0,
    'human.pedestrian.construction_worker': 0,
    'human.pedestrian.personal_mobility': 0,
    'human.pedestrian.police_officer': 0,
    'human.pedestrian.stroller': 0,
    'human.pedestrian.wheelchair': 0,
    'vehicle.bicycle': 1,
    'vehicle.car': 2,
    'vehicle.motorcycle': 3,
    'vehicle.bus.bendy': 5,
    'vehicle.bus.rigid': 5,
    'vehicle.truck': 7,
    'vehicle.emergency.ambulance': 2, # mapped to car
    'vehicle.emergency.police': 2, # mapped to car
    # Skip categories not in COCO80
    'animal': -1,
    'movable_object.barrier': -1,
    'movable_object.debris': -1,
    'movable_object.pushable_pullable': -1,
    'movable_object.trafficcone': -1,
    'static_object.bicycle_rack': -1,
    'vehicle.construction': -1,
    'vehicle.ego': -1,
    'vehicle.trailer': -1,
}  

def create_nuimages_output_dirs():
    for split in ['train', 'val', 'test', 'mini']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

def load_categories(split):
    category_file = os.path.join(nuimages_dir, 'labels', f'v1.0-{split}', 'category.json')
    if not os.path.exists(category_file):
        print(f"Warning: Category file not found: {category_file}")
        return {}
    
    with open(category_file, 'r') as f:
        categories = json.load(f)
    
    token_to_name = {}
    for cat in categories:
        token_to_name[cat['token']] = cat['name']
    return token_to_name

def load_sample_data(split):
    sample_data_file = os.path.join(nuimages_dir, 'labels', f'v1.0-{split}', 'sample_data.json')
    if not os.path.exists(sample_data_file):
        print(f"Warning: Sample data file not found: {sample_data_file}")
        return {}
    
    with open(sample_data_file, 'r') as f:
        sample_data = json.load(f)
    
    token_to_filename = {}
    for sample in sample_data:
        if sample.get('is_key_frame', False):
            token_to_filename[sample['token']] = sample['filename']
    return token_to_filename
def process_split(split):
    print(f"Processing {split} split...")
    
    token_to_category = load_categories(split)
    token_to_filename = load_sample_data(split)
    
    if not token_to_category or not token_to_filename:
        print(f"Skipping {split} due to missing metadata files")
        return
    
    filename_to_token = {filename: token for token, filename in token_to_filename.items()}
    
    samples_dir = os.path.join(nuimages_dir, 'samples')
    if not os.path.exists(samples_dir):
        print(f"Warning: Samples directory not found: {samples_dir}")
        return
    
    available_images = []
    for root, _, files in os.walk(samples_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                rel_path = os.path.relpath(os.path.join(root, file), samples_dir)
                full_rel_path = f"samples/{rel_path}"
                if full_rel_path in filename_to_token:
                    available_images.append(rel_path)
    
    print(f"Found {len(available_images)} available images for {split}")
    
    object_ann_file = os.path.join(nuimages_dir, 'labels', f'v1.0-{split}', 'object_ann.json')
    if not os.path.exists(object_ann_file):
        print(f"Warning: Object annotation file not found: {object_ann_file}")
        return
    
    with open(object_ann_file, 'r') as f:
        annotations = json.load(f)
    
    image_annotations = {}
    for ann in annotations:
        sample_token = ann['sample_data_token']
        if sample_token not in image_annotations:
            image_annotations[sample_token] = []
        image_annotations[sample_token].append(ann)
    
    print(f"Found annotations for {len(image_annotations)} images in {split}")
    
    output_images_dir = os.path.join(output_dir, 'images', split)
    output_labels_dir = os.path.join(output_dir, 'labels', split)
    
    processed = 0
    skipped = 0
    
    for image_filename in available_images:
        try:
            full_image_filename = f"samples/{image_filename}"
            sample_token = filename_to_token[full_image_filename]
            image_path = os.path.join(nuimages_dir, 'samples', image_filename)
            
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                skipped += 1
                continue
            
            with Image.open(image_path) as img:
                img_width, img_height = img.size
            
            image_basename = os.path.basename(image_filename)
            output_image_path = os.path.join(output_images_dir, image_basename)
            if not os.path.exists(output_image_path):
                shutil.copy2(image_path, output_image_path)
            
            anns = image_annotations.get(sample_token, [])
            yolo_annotations = []
            
            for ann in anns:
                category_token = ann['category_token']
                if category_token not in token_to_category:
                    continue
                
                category_name = token_to_category[category_token]
                if category_name not in NUIMAGES_TO_COCO80 or NUIMAGES_TO_COCO80[category_name] == -1:
                    continue
                
                bbox = ann['bbox']
                if not bbox or len(bbox) != 4:
                    continue
                
                x1, y1, x2, y2 = bbox
                if x1 >= x2 or y1 >= y2:
                    continue
                
                x_center, y_center, width, height = convert_bbox_to_yolo(
                    [x1, y1, x2, y2], img_width, img_height
                )
                if x_center < 0 or x_center > 1 or y_center < 0 or y_center > 1:
                    continue
                if width <= 0 or height <= 0 or width > 1 or height > 1:
                    continue
                
                class_id = NUIMAGES_TO_COCO80[category_name]
                yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                yolo_annotations.append(yolo_line)
            
            label_basename = os.path.splitext(image_basename)[0] + '.txt'
            label_path = os.path.join(output_labels_dir, label_basename)
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
            
            processed += 1
            if processed % 1000 == 0:
                print(f"Processed {processed} images for {split}")
                
        except Exception as e:
            print(f"Error processing image {image_filename}: {e}")
            skipped += 1
            continue
    
    print(f"Completed {split}: {processed} processed, {skipped} skipped")
def main():
    print("Converting nuImages to YOLO format with COCO80 labels...")
    
    if not os.path.exists(nuimages_dir):
        print(f"Error: nuImages directory not found: {nuimages_dir}")
        return
    
    create_nuimages_output_dirs()
    
    available_splits = ['train', 'val', 'test', 'mini']
    for split in available_splits:
        labels_dir = os.path.join(nuimages_dir, 'labels', f'v1.0-{split}')
        if os.path.exists(labels_dir):
            process_split(split)
        else:
            print(f"Warning: {split} split not found, skipping...")
    
    create_yaml_config(output_dir)
    print(f"Conversion complete! Dataset saved to: {output_dir}")

if __name__ == "__main__":
    main()