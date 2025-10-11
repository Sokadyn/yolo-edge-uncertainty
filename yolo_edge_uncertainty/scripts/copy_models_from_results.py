#!/usr/bin/env python3
"""
Copy all trained best.pt model files from
results/detect/data_splits_and_models/<train-...>/<model>/weights/best.pt
to yolo_edge_uncertainty/models/<train-...>/<model>/weights/best.pt,
preserving the path from the first train-* component. Overwrites existing files.

No arguments, no dry run.
"""
from pathlib import Path
import shutil
import sys


def main() -> int:
    # repo_root / yolo_edge_uncertainty / scripts / this_file
    repo_root = Path(__file__).resolve().parents[2]

    src = repo_root / "results" / "detect" / "data_splits_and_models"
    dst_root = repo_root / "yolo_edge_uncertainty" / "models"

    if not src.exists() or not src.is_dir():
        print(f"Source directory not found: {src}")
        return 1

    dst_root.mkdir(parents=True, exist_ok=True)

    # Find all .../weights/best.pt under any train-*/ model directories
    candidates = sorted(src.rglob("weights/best.pt"))
    best_files = []
    for f in candidates:
        try:
            rel = f.relative_to(src)
        except Exception:
            continue
        # Only keep paths that start with a train-* directory
        if len(rel.parts) >= 1 and rel.parts[0].startswith("train-"):
            best_files.append((f, rel))

    if not best_files:
        print("No best.pt files found to copy.")
        return 0

    copied = 0
    for src_path, rel in best_files:
        dst_path = dst_root / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src_path, dst_path)
            print(f"Copied: {src_path} -> {dst_path}")
            copied += 1
        except Exception as e:
            print(f"Failed to copy {src_path} -> {dst_path}: {e}")

    print(f"Done. Copied {copied} file(s) into {dst_root}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

