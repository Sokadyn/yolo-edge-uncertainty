import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ultralytics import settings
settings.update({"sync": False, "runs_dir": "interim_results"})
from ultralytics import YOLOEdgeUncertainty, YOLO
import torch
import datetime
import json
import pandas as pd
from IPython.display import display
import shutil

script_dir = os.path.dirname(os.path.abspath(__file__))


# for easier debugging
normal_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: f"{self.shape}_{normal_repr(self)}"

epochs = 25
fraction = 1.0
device = '0'
imgsz = 320 * 2
training = True # only run validation and load pretrained models
val_during_training = False # validation after each epoch during training for extended results

models = {
    "base-pretrained": YOLOEdgeUncertainty('yolo11n-base-confidence.yaml'),
    "base-confidence": YOLOEdgeUncertainty('yolo11n-base-confidence.yaml'),
    "base-uncertainty": YOLOEdgeUncertainty('yolo11n-base-uncertainty.yaml'),
    "ensemble": YOLOEdgeUncertainty('yolo11n-ensemble.yaml'),
    "mc-dropout": YOLOEdgeUncertainty('yolo11n-mc-dropout.yaml'),
    "edl-meh": YOLOEdgeUncertainty('yolo11n-edl-meh.yaml'),
}

folder_name = f"{script_dir}/../interim_results/detect/data_splits_and_models"

def update_results_csv(results, path, name):

    df = pd.DataFrame()
    for k, v in results.results_dict.items():
        df[k] = [v]

    df["speed_preprocess"] = results.speed["preprocess"]
    df["speed_inference"] = results.speed["inference"]
    df["speed_loss"] = results.speed["loss"]
    df["speed_postprocess"] = results.speed["postprocess"]
    df["name"] = name
    os.makedirs(path, exist_ok=True)
    df.to_csv(os.path.join(path, 'results_extended.csv'), index=False)
    return df

for name, model in models.items():

    train_dataset = 'cityscapes-train-kitti-val-from-coco80.yaml'
    train_dataset_name = os.path.splitext(os.path.basename(train_dataset))[0]
    print(f"Training model {name} on {train_dataset_name}")
    train_folder_name = f"{folder_name}/train-{train_dataset_name}/yolo_{name}"

    if name == "base-pretrained":
        model.load('yolo11n.pt')
    else:
        if training:
            # start from a clean folder
            if os.path.exists(train_folder_name):
                shutil.rmtree(train_folder_name)
            freeze_layers = [str(x) for x in range(23)] + ["23.cv2"] + ["23.dfl"]
            model.load('yolo11n.pt')
            train_results = model.train(data=train_dataset, epochs=epochs, verbose=False, plots=True, val=val_during_training, imgsz=imgsz, exist_ok=True, name=train_folder_name, device=device, fraction=fraction, rect=True, freeze=freeze_layers, box=0.0, dfl=0.0, cls=1.0)
            df_train = update_results_csv(train_results, train_folder_name, name)
            display(df_train)
        else:
            model.load(os.path.join(train_folder_name, 'weights', 'best.pt'))

    for val_dataset in ['cityscapes-from-coco80.yaml', 'foggy-cityscapes-from-coco80.yaml', 'rainy-cityscapes-from-coco80.yaml', 'kitti-from-coco80.yaml']:
        val_dataset_name = os.path.splitext(os.path.basename(val_dataset))[0]
        print(f"Validating model {name} on {val_dataset_name}")
        val_folder_name = train_folder_name.replace(f'train-{train_dataset_name}', f"val-{val_dataset_name}")
        if os.path.exists(val_folder_name):
            shutil.rmtree(val_folder_name)
        val_results = model.val(data=val_dataset, imgsz=imgsz, name=val_folder_name, exist_ok=True, device=device, fraction=fraction, rect=True)
        df_val = update_results_csv(val_results, val_folder_name, name)
        display(df_val)
