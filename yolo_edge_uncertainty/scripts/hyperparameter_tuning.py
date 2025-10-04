import sys
import os
import shutil

script_dir = os.path.dirname(os.path.abspath(__file__))
from ultralytics import settings
from ultralytics import YOLOEdgeUncertainty, YOLO
import torch
from ray import tune
import pandas as pd

from ultralytics.cfg import TASK2METRIC
TASK2METRIC["detect"] = "metrics/mUE50" # optimze for minmum uncertainty error during tuning

normal_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: f"{self.shape}_{normal_repr(self)}"
settings.update({"sync": False, "runs_dir": "results"})

def run_tuning(method_name: str, model_yaml: str, search_space: dict):
    """Run Ray Tune for a given method/model and save CSV summaries."""
    model = YOLOEdgeUncertainty(model_yaml)
    model.load('yolo11n.pt')

    folder_name = os.path.join(script_dir, "../../results/tuning", method_name)
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    result_grid = model.tune(
        name=folder_name,
        exist_ok=True,
        data="cityscapes-train-kitti-val-from-coco80-tuning.yaml",
        epochs=12,
        iterations=150,
        imgsz=320 * 2,
        space=search_space,
        use_ray=True,
        rect=False,
        save=False,
        fraction=1.0,
        grace_period=3,
        gpu_per_trial=0.1,
        val=False,
        freeze=[str(x) for x in range(23)] + ["23.cv2", "23.dfl"],
        box=0.0,
        dfl=0.0,
        cls=1.0
    )

    # Save all results
    path = result_grid.experiment_path
    df = result_grid.get_dataframe()
    results_csv_path = os.path.join(path, "results.csv")
    os.makedirs(path, exist_ok=True)
    df.to_csv(results_csv_path, index=False)
    print(f"[{method_name}] Saved all results to {results_csv_path}")

    result_mue = result_grid.get_best_result(metric="metrics/mUE50-95", mode="min")
    row_mue = result_mue.metrics.copy()
    row_mue['path'] = result_mue.path
    row_mue['filesystem'] = result_mue.filesystem
    row_mue['checkpoint'] = result_mue.checkpoint
    df_best_mue = pd.DataFrame([row_mue])
    results_best_mue_csv_path = os.path.join(path, "results_best_mue50-95.csv")
    df_best_mue.to_csv(results_best_mue_csv_path, index=False)
    print(f"[{method_name}] Saved best mUE50-95 result to {results_best_mue_csv_path}")

    result_map = result_grid.get_best_result(metric="metrics/mAP50-95(B)", mode="max")
    row_map = result_map.metrics.copy()
    row_map['path'] = result_map.path
    row_map['filesystem'] = result_map.filesystem
    row_map['checkpoint'] = result_map.checkpoint
    df_best_map = pd.DataFrame([row_map])
    results_best_map_csv_path = os.path.join(path, "results_best_map50-95B.csv")
    df_best_map.to_csv(results_best_map_csv_path, index=False)
    print(f"[{method_name}] Saved best mAP50-95(B) result to {results_best_map_csv_path}")

    # Calculate custom fitness metric
    df_with_fitness = df.copy()
    df_with_fitness['fitness_map50_plus_1_minus_mue50'] = (
        df_with_fitness['metrics/mAP50(B)'] + (1 - df_with_fitness['metrics/mUE50'])
    )
    
    best_fitness_idx = df_with_fitness['fitness_map50_plus_1_minus_mue50'].idxmax()
    best_fitness_row = df_with_fitness.loc[best_fitness_idx].to_dict()
    
    df_best_fitness = pd.DataFrame([best_fitness_row])
    results_best_fitness_csv_path = os.path.join(path, "results_best_map50_plus_1_minus_mue50.csv")
    df_best_fitness.to_csv(results_best_fitness_csv_path, index=False)
    print(f"[{method_name}] Saved best mAP50+(1-mUE50) result to {results_best_fitness_csv_path}")
    print(f"[{method_name}] Best fitness value: {best_fitness_row['fitness_map50_plus_1_minus_mue50']:.4f} "
          f"(mAP50: {best_fitness_row['metrics/mAP50(B)']:.4f}, mUE50: {best_fitness_row['metrics/mUE50']:.4f})")


ensemble_space = {
    #"ensemble_dropblock_size": tune.choice([3, 5, 7]),
    "ensemble_dropout_rate": tune.uniform(0.0, 0.5),
    "num_ensemble_heads": tune.choice([3, 5, 7, 9]),
}

mc_dropout_space = {
    #"mc_dropblock_size": tune.choice([3, 5, 7]),
    "mc_dropout_rate": tune.uniform(0.0, 0.5),
    "num_mc_forward_passes": tune.choice([5, 10, 15, 20]),
}

edl_meh_space = {
    #"meh_lambda_activation_idx": tune.choice([0, 1, 2, 3]),
    "edl_weight": tune.uniform(0.1, 5.0),
    "num_dirichlet_samples": tune.choice([5, 10, 15, 20])
}

if __name__ == "__main__":
    run_tuning("ensemble", 'yolo11n-ensemble.yaml', ensemble_space)
    run_tuning("mc-dropout", 'yolo11n-mc-dropout.yaml', mc_dropout_space)
    run_tuning("edl-meh", 'yolo11n-edl-meh.yaml', edl_meh_space)
