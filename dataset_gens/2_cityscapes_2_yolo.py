import os
import json
import shutil
from pathlib import Path
import yaml


script_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(os.path.dirname(script_dir), 'datasets')
cityscapes_dir = os.path.join(datasets_dir, 'cityscapes')
foggy_cityscapes_dir = os.path.join(datasets_dir, 'foggy_cityscapes')
rainy_cityscapes_dir = os.path.join(datasets_dir, 'rainy_cityscapes')

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
    """Loads a COCO JSON file."""
    with open(json_path, "r") as file:
        data = json.load(file)
    return data

train_data = load_coco_json(train_json)
val_data = load_coco_json(val_json)

class_mapping = {int(cat["id"]) - 1: cat["name"] for cat in train_data["categories"]} # COCO to YOLO class mapping

yaml_path = os.path.join(yolo_root, "labels/classes.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(class_mapping, f)

def convert_coco_to_yolo(annotations, images, yolo_label_dir, yolo_image_dir):
    """Converts COCO annotations to YOLO format and saves them."""
    image_id_to_filename = {img["id"]: img["file_name"] for img in images}
    image_id_to_size = {img["id"]: (img["width"], img["height"]) for img in images}

    yolo_annotations = {}
    for ann in annotations:
        img_id = ann["image_id"]
        if img_id not in image_id_to_filename:
            continue

        filename = image_id_to_filename[img_id]
        img_width, img_height = image_id_to_size[img_id]

        # Convert COCO bbox (x, y, w, h) -> YOLO (x_center, y_center, w, h)
        x, y, w, h = ann["bbox"]
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        width = w / img_width
        height = h / img_height

        class_id = ann["category_id"] - 1  # YOLO classes start at 0

        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        filename_without_dir = os.path.basename(filename)

        yolo_label_path = os.path.join(yolo_label_dir, filename_without_dir.replace(".png", ".txt"))
        if yolo_label_path not in yolo_annotations:
            yolo_annotations[yolo_label_path] = []
        yolo_annotations[yolo_label_path].append(yolo_line)

        src_image_path = os.path.join(image_root, filename)
        dst_image_path = os.path.join(yolo_image_dir, filename_without_dir)

        dest_dir = os.path.dirname(dst_image_path)
        os.makedirs(dest_dir, exist_ok=True)
        if not os.path.exists(dst_image_path):
            shutil.copy(src_image_path, dst_image_path)

    for label_path, lines in yolo_annotations.items():
        label_dir = os.path.dirname(label_path)
        os.makedirs(label_dir, exist_ok=True)
        with open(label_path, "w") as f:
            f.write("\n".join(lines))

convert_coco_to_yolo(train_data["annotations"], train_data["images"], yolo_label_dir_train, yolo_image_dir_train)
convert_coco_to_yolo(val_data["annotations"], val_data["images"], yolo_label_dir_val, yolo_image_dir_val)

# copy all test images over
test_image_dir = os.path.join(image_root, "leftImg8bit/test")
for city in os.listdir(test_image_dir):
    city_image_dir = os.path.join(test_image_dir, city)
    for img_file in os.listdir(city_image_dir):
        img_path = os.path.join(city_image_dir, img_file)
        shutil.copy(img_path, os.path.join(yolo_image_dir_test, img_file))

# go to foggy/rainy dir and copy all images from there and labels from cityscapes
for image_dir_src in [os.path.join(foggy_cityscapes_dir, "leftImg8bit_foggyDBF"), os.path.join(rainy_cityscapes_dir, "leftImg8bit_rain")]:
    dataset_name = image_dir_src.split("/")[-2]
    for split in ["train", "val"]:
        
        label_dir_src = yolo_label_dir_train if split == "train" else yolo_label_dir_val

        yolo_image_dir_dest = yolo_image_dir_train if split == "train" else yolo_image_dir_val
        yolo_image_dir_dest = yolo_image_dir_dest.replace("cityscapes", dataset_name)
        yolo_label_dir_dest = yolo_label_dir_train if split == "train" else yolo_label_dir_val
        yolo_label_dir_dest = yolo_label_dir_dest.replace("cityscapes", dataset_name)

        os.makedirs(yolo_image_dir_dest, exist_ok=True)
        os.makedirs(yolo_label_dir_dest, exist_ok=True)

        image_dir_src_split = os.path.join(image_dir_src, split)

        for city in os.listdir(image_dir_src_split):
            city_image_dir = os.path.join(image_dir_src_split, city)

            for img_file in Path(city_image_dir).glob("*.png"):
                img_filename = img_file.name
                img_dest = os.path.join(yolo_image_dir_dest, img_filename)

                if not os.path.exists(img_dest):
                    shutil.copy(img_file, img_dest)

                # example:
                # aachen_000000_000019_leftImg8bit.png
                # aachen_000000_000019_leftImg8bit_foggy_beta_0.01.png
                # aachen_000004_000019_leftImg8bit_rain_alpha_0.01_beta_0.005_dropsize_0.01_pattern_1.png

                label_filename_src = img_filename.split("_leftImg8bit")[0] + "_leftImg8bit.txt"
                label_filename_dest = img_filename.replace(".png", ".txt")
                # copy label
                city_label_dir = Path(label_dir_src) # / city
                label_path_src = city_label_dir / label_filename_src
                label_path_dest = os.path.join(yolo_label_dir_dest, label_filename_dest)


                if os.path.exists(img_file) and os.path.exists(label_path_src):
                    if not os.path.exists(label_path_dest):
                        shutil.copy(label_path_src, label_path_dest)
                else:
                    print(f"❌ Skipping label copy for {img_filename} (no label found)")

    shutil.copy(yaml_path, os.path.join(yolo_root.replace("cityscapes", dataset_name), "labels/classes.yaml"))
