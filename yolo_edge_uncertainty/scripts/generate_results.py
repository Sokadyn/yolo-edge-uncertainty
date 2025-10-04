import sys
import os

from ultralytics import settings
settings.update({"sync": False, "runs_dir": "results"})
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

def model_configs():
    return {
        "base-confidence": {
            "yaml": 'yolo11n-base-confidence.yaml',
            "val_kwargs": {"uncertainty_method": "sigmoid-complement"}
        },
        "ensemble": {
            "yaml": 'yolo11n-ensemble.yaml',
            "val_kwargs": {"uncertainty_method": "softmax-entropy"}
        },
        "mc-dropout": {
            "yaml": 'yolo11n-mc-dropout.yaml',
            "val_kwargs": {"uncertainty_method": "softmax-entropy"}
        },
        "edl-meh": {
            "yaml": 'yolo11n-edl-meh.yaml',
            "val_kwargs": {"uncertainty_method": "softmax-entropy"}
        },
    }

folder_name = f"{script_dir}/../../results/detect/data_splits_and_models"

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

train_datasets = ['cityscapes-from-coco80.yaml', 'kitti-from-coco80.yaml']
val_datasets = ['raincityscapes-from-coco80.yaml', 'foggy-cityscapes-from-coco80.yaml', 'bdd100k-coco80.yaml', 'nuimages-coco80.yaml']

for train_dataset in train_datasets:
    train_dataset_name = os.path.splitext(os.path.basename(train_dataset))[0]
    models = model_configs()
    
    for name, model_config in models.items():
        model_yaml = model_config["yaml"]
        val_kwargs = model_config["val_kwargs"]
        
        print(f"Training model {name} on {train_dataset_name}")
        train_folder_name = f"{folder_name}/train-{train_dataset_name}/yolo_{name}"

        if name == "base-pretrained":
            model = YOLOEdgeUncertainty(model_yaml)
            model.load('yolo11n.pt')
        else:
            if training:
                # start from a clean folder
                if os.path.exists(train_folder_name):
                    shutil.rmtree(train_folder_name)
                freeze_layers = [str(x) for x in range(23)] + ["23.cv2"] + ["23.dfl"]
                model = YOLOEdgeUncertainty(model_yaml)
                model.load('yolo11n.pt')
                train_results = model.train(
                    data=train_dataset,
                    epochs=epochs,
                    verbose=False,
                    plots=True,
                    val=val_during_training,
                    imgsz=imgsz,
                    exist_ok=True,
                    name=train_folder_name,
                    device=device,
                    fraction=fraction,
                    rect=False,
                    freeze=freeze_layers,
                    box=0.0,
                    dfl=0.0,
                    cls=1.0,
                )
                df_train = update_results_csv(train_results, train_folder_name, name)
                display(df_train)
            model = YOLOEdgeUncertainty(os.path.join(train_folder_name, 'weights', 'best.pt'))

        for val_dataset in val_datasets:
            val_dataset_name = os.path.splitext(os.path.basename(val_dataset))[0]
            print(f"Validating model {name} trained on {train_dataset_name} against {val_dataset_name}")
            val_folder_name = f"{folder_name}/train-{train_dataset_name}/val-{val_dataset_name}/yolo_{name}"
            if os.path.exists(val_folder_name):
                shutil.rmtree(val_folder_name)
            val_results = model.val(data=val_dataset, imgsz=imgsz, name=val_folder_name, exist_ok=True, device=device, rect=False, **val_kwargs)
            df_val = update_results_csv(val_results, val_folder_name, name)
            display(df_val)
