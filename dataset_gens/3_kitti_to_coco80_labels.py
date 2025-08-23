"""
Translates original KITTI dataset labels (already in YOLO format) to COCO 80 class IDs.

Usage:
    python3 3_kitti_to_coco80_labels.py

Requirements:
    - KITTI dataset should be located at '../datasets/kitti' and have the YOLO format

Output:
    - Creates a new directory '../datasets/kitti_from_coco80' containing updated labels and copied images.
"""

import os
import shutil
import yaml
from utils import COCO80_CLASSES


original_to_coco80_mapping = {
    0: 2,  # Car -> car
    1: 0,  # Pedestrian -> person
    2: 2,  # Van -> car
    3: 0,  # Cyclist -> person
    4: 7,  # Truck -> truck
    5: None,  # Misc -> (no direct mapping)
    6: 6,  # Tram -> train
    7: 0   # Person_sitting -> person
}


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
kitti_root = os.path.join(root_dir, "datasets", "kitti")

# Source paths (original KITTI structure)
yolo_images_train = os.path.join(kitti_root, "images", "train")
yolo_labels_train = os.path.join(kitti_root, "labels", "train")
yolo_images_val = os.path.join(kitti_root, "images", "val")
yolo_labels_val = os.path.join(kitti_root, "labels", "val")

# Target paths (with OCO80 labels)
kitti_from_coco80_root = os.path.join(root_dir, "datasets", "kitti_from_coco80")
yolo_images_train_new = os.path.join(kitti_from_coco80_root, "images", "train")
yolo_labels_train_new = os.path.join(kitti_from_coco80_root, "labels", "train")
yolo_images_val_new = os.path.join(kitti_from_coco80_root, "images", "val")
yolo_labels_val_new = os.path.join(kitti_from_coco80_root, "labels", "val")

os.makedirs(yolo_images_train_new, exist_ok=True)
os.makedirs(yolo_labels_train_new, exist_ok=True)
os.makedirs(yolo_images_val_new, exist_ok=True)
os.makedirs(yolo_labels_val_new, exist_ok=True)


def convert_labels_with_new_mapping(label_dir_src, label_dir_dest, mapping):
    """Copies label files to a new directory and updates the class IDs using the mapping."""
    for label_file in os.listdir(label_dir_src):
        if label_file.endswith(".txt"):
            with open(os.path.join(label_dir_src, label_file), "r") as f:
                lines = f.readlines()

            updated_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    if class_id in mapping:
                        updated_class_id = mapping[class_id]
                        if updated_class_id is not None:
                            updated_line = f"{updated_class_id} {' '.join(parts[1:])}\n"
                            updated_lines.append(updated_line)

            dest_label_path = os.path.join(label_dir_dest, label_file)
            with open(dest_label_path, "w") as f:
                f.writelines(updated_lines)

# Copy images to the new directory
for filename in os.listdir(yolo_images_train):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        src_img_path = os.path.join(yolo_images_train, filename)
        dst_img_path = os.path.join(yolo_images_train_new, filename)
        if not os.path.exists(dst_img_path):
            shutil.copy(src_img_path, dst_img_path)
for filename in os.listdir(yolo_images_val):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        src_img_path = os.path.join(yolo_images_val, filename)
        dst_img_path = os.path.join(yolo_images_val_new, filename)
        if not os.path.exists(dst_img_path):
            shutil.copy(src_img_path, dst_img_path)

convert_labels_with_new_mapping(yolo_labels_train, yolo_labels_train_new, original_to_coco80_mapping)
convert_labels_with_new_mapping(yolo_labels_val, yolo_labels_val_new, original_to_coco80_mapping)

labels_dir_new = os.path.join(kitti_from_coco80_root, "labels")
os.makedirs(labels_dir_new, exist_ok=True)
with open(os.path.join(labels_dir_new, "classes.yaml"), "w") as f:
    yaml.dump(COCO80_CLASSES, f)

print("Dataset with updated labels has been created successfully.")
