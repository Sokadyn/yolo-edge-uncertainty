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

class_mapping_coco80 = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush'
}

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
    yaml.dump(class_mapping_coco80, f)

print("Dataset with updated labels has been created successfully.")
