"""
Convert Cityscapes dataset annotations (already in YOLO format) to COCO80 labels.

Usage:
    python3 3_cityscapes_to_coco80_labels.py

Requirements:
    - Cityscapes dataset should be located at '../datasets/cityscapes' with YOLO format labels
    - Foggy Cityscapes dataset should be located at '../datasets/foggy_cityscapes' with YOLO format labels
    - RainCityscapes dataset should be located at '../datasets/raincityscapes' with YOLO format labels

Output:
    - Creates directories and organizes images and labels in COCO80 format at '../datasets/cityscapes_from_coco80'.
    - Creates similar directory structures for foggy and RainCityscapes datasets with updated labels.
"""

import os
import json
import shutil
from pathlib import Path
import yaml
from utils import COCO80_CLASSES

script_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(os.path.dirname(script_dir), 'datasets')
cityscapes_dir = os.path.join(datasets_dir, 'cityscapes')
foggy_cityscapes_dir = os.path.join(datasets_dir, 'foggy_cityscapes')
raincityscapes_dir = os.path.join(datasets_dir, 'raincityscapes')

coco_annotations_dir = os.path.join(cityscapes_dir, 'annotations')
image_root = cityscapes_dir # /leftImg8bit already in json annotations

yolo_root = cityscapes_dir
yolo_image_dir_train = os.path.join(yolo_root, "images/train")
yolo_image_dir_val = os.path.join(yolo_root, "images/val")
yolo_image_dir_test = os.path.join(yolo_root, "images/test")
yolo_label_dir_train = os.path.join(yolo_root, "labels/train")
yolo_label_dir_val = os.path.join(yolo_root, "labels/val")

os.makedirs(yolo_image_dir_train, exist_ok=True)
os.makedirs(yolo_image_dir_val, exist_ok=True)
os.makedirs(yolo_image_dir_test, exist_ok=True)
os.makedirs(yolo_label_dir_train, exist_ok=True)
os.makedirs(yolo_label_dir_val, exist_ok=True)

train_json = os.path.join(coco_annotations_dir, "instancesonly_filtered_gtFine_train.json")
val_json = os.path.join(coco_annotations_dir, "instancesonly_filtered_gtFine_val.json")

def load_coco_json(json_path):
    with open(json_path, "r") as file:
        data = json.load(file)
    return data

train_data = load_coco_json(train_json)
val_data = load_coco_json(val_json)

original_class_mapping = {
    0: 'person',
    1: 'rider',
    2: 'car',
    3: 'motorcycle',
    4: 'bicycle',
    5: 'bus',
    6: 'truck',
    7: 'train'
}


original_to_coco80_mapping = {
    0: 0,  # person -> person
    1: 0,  # rider -> person
    2: 2,  # car -> car
    3: 3,  # motorcycle -> motorcycle
    4: 1,  # bicycle -> bicycle
    5: 5,  # bus -> bus
    6: 7,  # truck -> truck
    7: 6   # train -> train
}

cityscapes_from_coco80_root = os.path.join(datasets_dir, 'cityscapes_from_coco80')

new_image_dir_train = os.path.join(cityscapes_from_coco80_root, "images/train")
new_image_dir_val = os.path.join(cityscapes_from_coco80_root, "images/val")
new_image_dir_test = os.path.join(cityscapes_from_coco80_root, "images/test")
new_label_dir_train = os.path.join(cityscapes_from_coco80_root, "labels/train")
new_label_dir_val = os.path.join(cityscapes_from_coco80_root, "labels/val")

os.makedirs(new_image_dir_train, exist_ok=True)
os.makedirs(new_image_dir_val, exist_ok=True)
os.makedirs(new_image_dir_test, exist_ok=True)
os.makedirs(new_label_dir_train, exist_ok=True)
os.makedirs(new_label_dir_val, exist_ok=True)

foggy_image_dir_train = os.path.join(foggy_cityscapes_dir, "images", "train")
foggy_image_dir_val = os.path.join(foggy_cityscapes_dir, "images", "val")
rainy_image_dir_train = os.path.join(raincityscapes_dir, "images", "train")
rainy_image_dir_val = os.path.join(raincityscapes_dir, "images", "val")

foggy_label_dir_train = os.path.join(foggy_cityscapes_dir, "labels", "train")
foggy_label_dir_val = os.path.join(foggy_cityscapes_dir, "labels", "val")
rainy_label_dir_train = os.path.join(raincityscapes_dir, "labels", "train")
rainy_label_dir_val = os.path.join(raincityscapes_dir, "labels", "val")

foggy_cityscapes_from_coco80 = os.path.join(datasets_dir, 'foggy_cityscapes_from_coco80')
raincityscapes_from_coco80 = os.path.join(datasets_dir, 'raincityscapes_from_coco80')

new_foggy_image_dir_train = os.path.join(foggy_cityscapes_from_coco80, "images", "train")
new_foggy_image_dir_val = os.path.join(foggy_cityscapes_from_coco80, "images", "val")
new_rainy_image_dir_train = os.path.join(raincityscapes_from_coco80, "images", "train")
new_rainy_image_dir_val = os.path.join(raincityscapes_from_coco80, "images", "val")

new_foggy_label_dir_train = os.path.join(foggy_cityscapes_from_coco80, "labels", "train")
new_foggy_label_dir_val = os.path.join(foggy_cityscapes_from_coco80, "labels", "val")
new_rainy_label_dir_train = os.path.join(raincityscapes_from_coco80, "labels", "train")
new_rainy_label_dir_val = os.path.join(raincityscapes_from_coco80, "labels", "val")

os.makedirs(new_foggy_image_dir_train, exist_ok=True)
os.makedirs(new_foggy_image_dir_val, exist_ok=True)
os.makedirs(new_rainy_image_dir_train, exist_ok=True)
os.makedirs(new_rainy_image_dir_val, exist_ok=True)

os.makedirs(new_foggy_label_dir_train, exist_ok=True)
os.makedirs(new_foggy_label_dir_val, exist_ok=True)
os.makedirs(new_rainy_label_dir_train, exist_ok=True)
os.makedirs(new_rainy_label_dir_val, exist_ok=True)

def update_and_copy_labels(label_dir_src, label_dir_dest, mapping):
    """Copies label files to a new directory and updates the class IDs using the mapping."""
    for label_file in Path(label_dir_src).glob("*.txt"):
        with open(label_file, "r") as f:
            lines = f.readlines()

        updated_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                if class_id in mapping:
                    updated_class_id = mapping[class_id]
                    updated_line = f"{updated_class_id} {' '.join(parts[1:])}\n"
                    updated_lines.append(updated_line)

        dest_label_path = os.path.join(label_dir_dest, label_file.name)
        with open(dest_label_path, "w") as f:
            f.writelines(updated_lines)

shutil.copytree(yolo_image_dir_train, new_image_dir_train, dirs_exist_ok=True)
shutil.copytree(yolo_image_dir_val, new_image_dir_val, dirs_exist_ok=True)
update_and_copy_labels(yolo_label_dir_train, new_label_dir_train, original_to_coco80_mapping)
update_and_copy_labels(yolo_label_dir_val, new_label_dir_val, original_to_coco80_mapping)

shutil.copytree(foggy_image_dir_train, new_foggy_image_dir_train, dirs_exist_ok=True)
shutil.copytree(foggy_image_dir_val, new_foggy_image_dir_val, dirs_exist_ok=True)
update_and_copy_labels(foggy_label_dir_train, new_foggy_label_dir_train, original_to_coco80_mapping)
update_and_copy_labels(foggy_label_dir_val, new_foggy_label_dir_val, original_to_coco80_mapping)

shutil.copytree(rainy_image_dir_train, new_rainy_image_dir_train, dirs_exist_ok=True)
shutil.copytree(rainy_image_dir_val, new_rainy_image_dir_val, dirs_exist_ok=True)
update_and_copy_labels(rainy_label_dir_train, new_rainy_label_dir_train, original_to_coco80_mapping)
update_and_copy_labels(rainy_label_dir_val, new_rainy_label_dir_val, original_to_coco80_mapping)

shutil.copytree(yolo_image_dir_test, new_image_dir_test, dirs_exist_ok=True)

new_dir = 'cityscapes_from_coco80'
labels_dir = os.path.join(cityscapes_from_coco80_root, "labels")
yaml_file_path = os.path.join(labels_dir, 'classes.yaml')
os.makedirs(labels_dir, exist_ok=True)

with open(yaml_file_path, 'w') as file:
    yaml.dump(COCO80_CLASSES, file)

foggy_yaml_file_path = os.path.join(foggy_cityscapes_from_coco80, "labels", 'classes.yaml')
with open(foggy_yaml_file_path, 'w') as file:
    yaml.dump(COCO80_CLASSES, file)
rainy_yaml_file_path = os.path.join(raincityscapes_from_coco80, "labels", 'classes.yaml')
with open(rainy_yaml_file_path, 'w') as file:
    yaml.dump(COCO80_CLASSES, file)

print("Dataset with updated labels has been created successfully.")
