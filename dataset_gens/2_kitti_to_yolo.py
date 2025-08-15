"""
Convert KITTI dataset annotations to YOLO format and organize them into training and validation directories.

Usage:
    python3 2_kitti_to_yolo.py

Requirements:
    - The KITTI dataset must be located in '../datasets/kitti'.

Output:
    - The script will create directories for training and validation images and labels within the KITTI dataset directory.

References:
    - https://gist.github.com/nikvaessen/20f29642cc23a1f45f41e7cc507acb7a#file-split1_trainl-txt
"""

import cv2
import yaml
import os
import shutil
import urllib.request

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Source paths (original KITTI structure)
kitti_root = os.path.join(root_dir, "datasets", "kitti")
image_dir = os.path.join(kitti_root, "images", "training", "image_2")
label_dir = os.path.join(kitti_root, "labels", "training", "label_2")

# Output paths (YOLO structure for training and validation)
yolo_images_train = os.path.join(kitti_root, "images", "train")
yolo_labels_train = os.path.join(kitti_root, "labels", "train")
yolo_images_val = os.path.join(kitti_root, "images", "val")
yolo_labels_val = os.path.join(kitti_root, "labels", "val")

os.makedirs(yolo_images_train, exist_ok=True)
os.makedirs(yolo_labels_train, exist_ok=True)
os.makedirs(yolo_images_val, exist_ok=True)
os.makedirs(yolo_labels_val, exist_ok=True)

class_mapping = {
    'Car': 0,
    'Pedestrian': 1,
    'Van': 2,
    'Cyclist': 3,
    'Truck': 4,
    'Misc': 5,
    'Tram': 6,
    'Person_sitting': 7
}

class_mapping_r = {v: k for k, v in class_mapping.items()}

labels_dir = os.path.join(kitti_root, "labels")
os.makedirs(labels_dir, exist_ok=True)
with open(os.path.join(labels_dir, "classes.yaml"), "w") as f:
    yaml.dump(class_mapping_r, f)

def convert_kitti_to_yolo(kitti_label_path, yolo_label_path, image_width, image_height):
    with open(kitti_label_path, "r") as file:
        lines = file.readlines()
    yolo_lines = []
    for line in lines:
        parts = line.strip().split()
        class_name, x1, y1, x2, y2 = (
            parts[0], float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
        )
        if class_name not in class_mapping:
            continue
        class_id = class_mapping[class_name]
        x_center = ((x1 + x2) / 2) / image_width
        y_center = ((y1 + y2) / 2) / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    if yolo_lines:
        with open(yolo_label_path, "w") as f:
            f.write("\n".join(yolo_lines))

def process_split_files(split_file_url, image_dest_dir, label_dest_dir):
    with urllib.request.urlopen(split_file_url) as response:
        split_content = response.read().decode('utf-8')
    filenames = split_content.splitlines()

    for filename in filenames:
        src_img_path = os.path.join(image_dir, filename + ".png")
        dst_img_path = os.path.join(image_dest_dir, filename + ".png")
        if not os.path.exists(dst_img_path):
            shutil.copy(src_img_path, dst_img_path)

        img = cv2.imread(src_img_path)
        if img is None:
            print(f"Could not read {src_img_path}")
            continue

        image_width, image_height = img.shape[1], img.shape[0]
        label_filename = filename + ".txt"
        kitti_label_path = os.path.join(label_dir, label_filename)
        yolo_label_path = os.path.join(label_dest_dir, label_filename)

        if os.path.exists(kitti_label_path):
            convert_kitti_to_yolo(kitti_label_path, yolo_label_path, image_width, image_height)

train_split_url = "https://gist.github.com/nikvaessen/20f29642cc23a1f45f41e7cc507acb7a/raw/4dac16a03220c1a2f60bfb15b3ba156e9f48acc7/split1_trainl.txt"
val_split_url = "https://gist.github.com/nikvaessen/20f29642cc23a1f45f41e7cc507acb7a/raw/4dac16a03220c1a2f60bfb15b3ba156e9f48acc7/split1_val.txt"

print("Processing training split...")
process_split_files(train_split_url, yolo_images_train, yolo_labels_train)

print("Processing validation split...")
process_split_files(val_split_url, yolo_images_val, yolo_labels_val)

print("Processing completed.")
