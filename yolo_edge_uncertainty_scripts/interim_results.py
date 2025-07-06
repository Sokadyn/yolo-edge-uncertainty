import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ultralytics import settings
settings.update({"sync": False, "runs_dir": "interim_results", "datasets_dir": "datasets"})
from ultralytics import YOLOEdgeUncertainty, YOLO
import torch
import datetime
import json
import pandas as pd
from IPython.display import display

script_dir = os.path.dirname(os.path.abspath(__file__))

# for easier debugging
normal_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: f"{self.shape}_{normal_repr(self)}"

epochs = 50
fraction = 1.0
device = '0'
imgsz = 320

models = {
    "base-confidence": YOLOEdgeUncertainty('yolo11n-base-confidence.yaml'),
    "base-uncertainty": YOLOEdgeUncertainty('yolo11n-base-uncertainty.yaml'),
    "ensemble": YOLOEdgeUncertainty('yolo11n-ensemble.yaml'),
    "mc-dropout": YOLOEdgeUncertainty('yolo11n-mc-dropout.yaml'),
    "edl-meh": YOLOEdgeUncertainty('yolo11n-edl-meh.yaml')
}

date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_name = f"{script_dir}/../interim_results/detect/{date}_models-{len(models)}_epochs-{epochs}"

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
    df.to_csv(os.path.join(path, 'results.csv'), index=False)
    return df

for name, model in models.items():
    print(f"Training model {name}")
    model.load('yolo11n.pt')
    train_folder_name = f"{folder_name}/train/yolo_{name}"
    train_results = model.train(data='cityscapes.yaml', epochs=epochs, verbose=False, plots=True, val=False, imgsz=imgsz, exist_ok=True, name=train_folder_name, device=device, fraction=fraction)
    df_train = update_results_csv(train_results, train_folder_name, name)
    display(df_train)

    print(f"Validating model {name}")
    val_folder_name = train_folder_name.replace('train', 'val')
    val_results = model.val(data='foggy-cityscapes.yaml', imgsz=imgsz, name=val_folder_name, exist_ok=True, device=device, fraction=fraction)
    df_val = update_results_csv(val_results, val_folder_name, name)
    display(df_val)



