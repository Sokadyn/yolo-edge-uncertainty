"""
Benchmark all Cityscapes-trained uncertainty models and aggregate results.

This script:
- Builds a dict of the 4 Cityscapes-trained models and their weight paths
- Benchmarks each using Ultralytics' benchmark utility (ONNX format by default)
- Aggregates results into a single DataFrame
- Saves the CSV under yolo_edge_uncertainty/csv/benchmark_inference_time.csv
"""

from pathlib import Path

import pandas as pd

from ultralytics.utils.benchmarks import benchmark
from ultralytics import YOLOEdgeUncertainty


def get_cityscapes_models_dict() -> dict:
    base = Path("yolo_edge_uncertainty/models/train-cityscapes-from-coco80")
    return {
        "yolo_base-confidence": base / "yolo_base-confidence" / "weights" / "best.pt",
        "yolo_mc-dropout": base / "yolo_mc-dropout" / "weights" / "best.pt",
        "yolo_ensemble": base / "yolo_ensemble" / "weights" / "best.pt",
        "yolo_edl-meh": base / "yolo_edl-meh" / "weights" / "best.pt",
    }


def run_benchmarks(models: dict, imgsz: int = 640, data: str = "raincityscapes.yaml", fmt: str = "openvino", device: str = "cpu") -> pd.DataFrame:
    rows = []
    for name, weights in models.items():
        print(f"Benchmarking {name}: {weights}")
        model = YOLOEdgeUncertainty(str(weights))
        df = benchmark(
            model=model,
            data=data,
            imgsz=imgsz,
            format=fmt,
            save=True,
            plots=True,
            device=device,
            nms=True,
            int8=True
        )
        df.insert(0, "Model", name)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def main():
    models = get_cityscapes_models_dict()
    results = run_benchmarks(models)
    out_dir = Path("yolo_edge_uncertainty/csv")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "benchmark_inference_time.csv"
    results.to_csv(out_file, index=False)
    print(f"Saved aggregated benchmarks to {out_file}")


if __name__ == "__main__":
    main()

