"""
Convert YOLO KITTI dataset labels to Cityscapes label ids, and copy images to a new directory structure.

Usage:
    python3 3_kitti_to_cityscapes_labels.py

Requirements:
    - KITTI dataset should be located at '../datasets/kitti' with YOLO format labels.

Output:
    - A new directory '../datasets/kitti_from_cityscapes' containing updated labels and copied images in Cityscapes format.
"""

import os
import shutil
import yaml

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cityscapes_class_names = [
    "person",      # 0
    "rider",       # 1
    "car",         # 2
    "motorcycle",  # 3
    "bicycle",     # 4
    "bus",         # 5
    "truck",       # 6
    "train",       # 7
]

kitti_to_cityscapes = {
    0: 2,    # Car -> car
    1: 0,    # Pedestrian -> person
    2: 6,    # Van -> truck
    3: 1,    # Cyclist -> rider
    4: 6,    # Truck -> truck
    5: None, # Misc -> (no direct mapping)
    6: 7,    # Tam -r> train
    7: 0     # Person_sitting -> person
}

# Input directories (YOLO KITTI format)
yolo_label_dir_train = os.path.join(root_dir, "datasets", "kitti", "labels", "train")
yolo_label_dir_val = os.path.join(root_dir, "datasets", "kitti", "labels", "val")
image_dir_train = os.path.join(root_dir, "datasets", "kitti", "images", "train")
image_dir_val = os.path.join(root_dir, "datasets", "kitti", "images", "val")

# Temporary mapping output
temp_label_path = os.path.join(root_dir, "datasets", "kitti", "labels_cityscapes")
cityscapes_label_dir_train = os.path.join(temp_label_path, "train")
cityscapes_label_dir_val = os.path.join(temp_label_path, "val")
os.makedirs(cityscapes_label_dir_train, exist_ok=True)
os.makedirs(cityscapes_label_dir_val, exist_ok=True)

def convert_yolo_to_cityscapes(yolo_label_path, cityscapes_label_path):
    with open(yolo_label_path, "r") as file:
        lines = file.readlines()
    cityscapes_lines = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        cityscapes_class_id = kitti_to_cityscapes.get(class_id, None)
        if cityscapes_class_id is not None:
            cityscapes_lines.append(f"{cityscapes_class_id} {' '.join(parts[1:])}")
    if cityscapes_lines:
        with open(cityscapes_label_path, "w") as f:
            f.write("\n".join(cityscapes_lines))

print("Converting YOLO labels to Cityscapes format...")
for filename in sorted(os.listdir(yolo_label_dir_train)):
    if filename.endswith(".txt"):
        yolo_label_path = os.path.join(yolo_label_dir_train, filename)
        cityscapes_label_path = os.path.join(cityscapes_label_dir_train, filename)
        convert_yolo_to_cityscapes(yolo_label_path, cityscapes_label_path)

for filename in sorted(os.listdir(yolo_label_dir_val)):
    if filename.endswith(".txt"):
        yolo_label_path = os.path.join(yolo_label_dir_val, filename)
        cityscapes_label_path = os.path.join(cityscapes_label_dir_val, filename)
        convert_yolo_to_cityscapes(yolo_label_path, cityscapes_label_path)


new_root = os.path.join(root_dir, "datasets", "kitti_from_cityscapes")
new_images_dir_train = os.path.join(new_root, "images", "train")
new_labels_dir_train = os.path.join(new_root, "labels", "train")
os.makedirs(new_images_dir_train, exist_ok=True)
os.makedirs(new_labels_dir_train, exist_ok=True)

new_images_dir_val = os.path.join(new_root, "images", "val")
new_labels_dir_val = os.path.join(new_root, "labels", "val")
os.makedirs(new_images_dir_val, exist_ok=True)
os.makedirs(new_labels_dir_val, exist_ok=True)

print("Copying images to kitti_from_cityscapes...")
for filename in sorted(os.listdir(image_dir_train)):
    src_path = os.path.join(image_dir_train, filename)
    dst_path = os.path.join(new_images_dir_train, filename)
    if os.path.isfile(src_path) and not os.path.exists(dst_path):
        shutil.copy(src_path, dst_path)

for filename in sorted(os.listdir(image_dir_val)):
    src_path = os.path.join(image_dir_val, filename)
    dst_path = os.path.join(new_images_dir_val, filename)
    if os.path.isfile(src_path) and not os.path.exists(dst_path):
        shutil.copy(src_path, dst_path)

print("Copying mapped labels to kitti_from_cityscapes...")
for filename in sorted(os.listdir(cityscapes_label_dir_train)):
    src_path = os.path.join(cityscapes_label_dir_train, filename)
    dst_path = os.path.join(new_labels_dir_train, filename)
    if os.path.isfile(src_path) and not os.path.exists(dst_path):
        shutil.copy(src_path, dst_path)

for filename in sorted(os.listdir(cityscapes_label_dir_val)):
    src_path = os.path.join(cityscapes_label_dir_val, filename)
    dst_path = os.path.join(new_labels_dir_val, filename)
    if os.path.isfile(src_path) and not os.path.exists(dst_path):
        shutil.copy(src_path, dst_path)


class_dict = {i: name for i, name in enumerate(cityscapes_class_names)}
classes_yaml_path = os.path.join(new_root, "labels", "classes.yaml")
with open(classes_yaml_path, "w") as f:
    yaml.dump(class_dict, f, default_flow_style=False, allow_unicode=True)

if os.path.exists(temp_label_path):
    shutil.rmtree(temp_label_path)

print("All images and mapped labels are now in datasets/kitti_from_cityscapes/.")
