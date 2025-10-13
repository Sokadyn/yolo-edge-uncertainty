#!/usr/bin/env python3
"""
Export all best.pt files under yolo_edge_uncertainty/models to best.onnx
in the same weights folder, overwriting if present. No arguments.
"""
from pathlib import Path
import os
import sys

from ultralytics import settings
settings.update({"sync": False, "runs_dir": "results"})
from ultralytics import YOLOEdgeUncertainty


def export_one(pt_path: Path, device: str = "0", imgsz: int = 640) -> Path:
    onnx_path = pt_path.with_suffix(".onnx")
    try:
        if onnx_path.exists():
            onnx_path.unlink()
    except Exception:
        pass

    model = YOLOEdgeUncertainty(str(pt_path))
    model.export(format="onnx", device=device, imgsz=imgsz, nms=True)
    return onnx_path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    models_root = repo_root / "yolo_edge_uncertainty" / "models"

    if not models_root.exists():
        print(f"Models directory not found: {models_root}")
        return 1

    settings_dir = repo_root / ".ultralytics"
    settings_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ULTRALYTICS_SETTINGS_DIR", str(settings_dir))

    pt_files = sorted(models_root.rglob("best.pt"))
    if not pt_files:
        print("No best.pt files found under models directory.")
        return 0

    count = 0
    for pt in pt_files:
        try:
            out = export_one(pt)
            print(f"Exported: {pt} -> {out}")
            count += 1
        except Exception as e:
            print(f"Failed to export {pt}: {e}")

    print(f"Done. Exported {count} model(s) to ONNX.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

