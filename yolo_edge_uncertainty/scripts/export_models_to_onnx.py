import os
import shutil
from pathlib import Path
device = '0'
from ultralytics import YOLOEdgeUncertainty
imgsz = 320 * 2

def model_configs():
    return {
        "base-confidence": {
            "yaml": 'yolo11n-base-confidence.yaml',
        },
        "ensemble": {
            "yaml": 'yolo11n-ensemble.yaml',
        },
        "mc-dropout": {
            "yaml": 'yolo11n-mc-dropout.yaml',
        },
        "edl-meh": {
            "yaml": 'yolo11n-edl-meh.yaml',
        },
    }


def export_to_onnx(weights_path: Path, export_imgsz: int, export_device: str | None = None) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    settings_dir = repo_root / ".ultralytics"
    settings_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ULTRALYTICS_SETTINGS_DIR", str(settings_dir))

    model = YOLOEdgeUncertainty(str(weights_path))
    out = model.export(format="onnx", imgsz=export_imgsz, device=export_device, nms=True)
    return out


def copy_train_models(folder_root: Path, dest_root: Path) -> tuple[int, int]:
    copied_pt = 0
    copied_onnx = 0

    if not folder_root.exists():
        return (0, 0)

    train_dirs = [d for d in folder_root.iterdir() if d.is_dir() and d.name.startswith("train-")]
    for train_dir in sorted(train_dirs):
        model_dirs = [m for m in train_dir.iterdir() if m.is_dir() and m.name.startswith("yolo_")]
        for mdir in sorted(model_dirs):
            wdir = mdir / "weights"
            if not wdir.exists():
                continue

            dest_wdir = dest_root / train_dir.name / mdir.name / "weights"
            dest_wdir.mkdir(parents=True, exist_ok=True)

            src_pt = wdir / "best.pt"
            if src_pt.exists():
                dest_pt = dest_wdir / src_pt.name
                shutil.copy2(src_pt, dest_pt)
                copied_pt += 1

            src_onnx = wdir / "best.onnx"
            if src_onnx.exists():
                dest_onnx = dest_wdir / src_onnx.name
                shutil.copy2(src_onnx, dest_onnx)
                copied_onnx += 1

    return (copied_pt, copied_onnx)

def main():
    script_dir = Path(__file__).resolve().parent
    folder_root = (script_dir / ".." / ".." / "results" / "detect" / "data_splits_and_models").resolve()
    dest_root = (script_dir / ".." / "models").resolve()

    train_datasets = ['cityscapes-from-coco80.yaml', 'kitti-from-coco80.yaml']

    cfgs = model_configs()

    exported_count = 0
    for train_dataset in train_datasets:
        train_dataset_name = Path(train_dataset).stem
        for name, cfg in cfgs.items():
            train_folder = folder_root / f"train-{train_dataset_name}" / f"yolo_{name}"
            best_pt = train_folder / "weights" / "best.pt"
            if not best_pt.exists():
                continue
            try:
                export_to_onnx(best_pt, export_imgsz=max(imgsz if isinstance(imgsz, (list, tuple)) else (imgsz, imgsz)), export_device=device)
                exported_count += 1
            except Exception:
                pass

    copied_pt, copied_onnx = copy_train_models(folder_root, dest_root)
    print(f"Copied {copied_pt} best.pt and {copied_onnx} best.onnx from {folder_root} to {dest_root}")


if __name__ == "__main__":
    main()
