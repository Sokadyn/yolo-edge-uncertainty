"""
Copy exported model artifacts into a central folder.
  Collect all `.pt` and `.onnx` files from training/validation results and the
  repository root, and copy them into `yolo_edge_uncertainty/models`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
from typing import Iterable, List, Tuple


DEFAULT_EXTS = {".pt", ".onnx"}

def discover_files(base: Path, exts: set[str]) -> List[Path]:
    """Recursively discover files with given extensions under `base`.

    Returns sorted unique paths.
    """
    if not base.exists():
        return []

    found: set[Path] = set()
    for root, _, files in os.walk(base):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in exts:
                found.add(p.resolve())
    return sorted(found)


def collision_safe_name(path: Path, relative_to: Path) -> str:
    """Create a collision-safe filename that encodes its relative path.

    Example:
    results/detect/data_splits_and_models/train-kitti/yolo_mc-dropout/weights/best.pt
    -> detect__data_splits_and_models__train-kitti__yolo_mc-dropout__weights__best.pt
    """
    try:
        rel = path.resolve().relative_to(relative_to.resolve())
    except Exception:
        rel = Path(path.parent.name) / path.name
    parts = list(rel.parts)
    if parts and parts[0] == "results":
        parts = parts[1:]
    return "__".join(parts)


def copy_files(
    files: Iterable[Path],
    dest_dir: Path,
    relative_to: Path,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """Copy files into `dest_dir` using collision-safe names.

    Returns (copied_count, skipped_count).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    for src in files:
        dst_name = collision_safe_name(src, relative_to)
        dst = dest_dir / dst_name
        if dst.exists() and not overwrite:
            skipped += 1
            print(f"[skip] {dst.name} already exists")
            continue
        print(("[dry-run] would copy" if dry_run else "copy"), src, "->", dst)
        if not dry_run:
            shutil.copy2(src, dst)
        copied += 1
    return copied, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        action="append",
        default=None,
        help=(
            "Source directory to scan (recursively). "
            "Can be specified multiple times. Defaults to: 'results' and repo root."
        ),
    )
    parser.add_argument(
        "--dest",
        default=None,
        help="Destination directory. Defaults to yolo_edge_uncertainty/models",
    )
    parser.add_argument(
        "--ext",
        action="append",
        default=None,
        help="File extension to include (e.g., .pt). Can be repeated. Defaults to .pt and .onnx",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite if destination exists")
    parser.add_argument("--dry-run", action="store_true", help="List operations without copying")

    args = parser.parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    repo_root = (script_dir / ".." / "..").resolve()

    src_dirs: List[Path]
    if args.src:
        src_dirs = [Path(s).resolve() for s in args.src]
    else:
        src_dirs = [repo_root / "results"]

    dest_dir = Path(args.dest).resolve() if args.dest else (repo_root / "yolo_edge_uncertainty" / "models").resolve()
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in (args.ext or [])}
    if not exts:
        exts = set(DEFAULT_EXTS)

    print("Sources:")
    for s in src_dirs:
        print(" -", s)
    print("Destination:", dest_dir)
    print("Extensions:", ", ".join(sorted(exts)))
    print()

    # Discover and copy
    all_files: List[Path] = []
    for src in src_dirs:
        files = discover_files(src, exts)
        # Make a stable order (shorter paths first for nicer output)
        files.sort(key=lambda p: (len(str(p)), str(p)))
        print(f"Found {len(files)} files under {src}")
        all_files.extend(files)

    # Deduplicate while preserving order
    seen: set[Path] = set()
    uniq_files: List[Path] = []
    for f in all_files:
        if f not in seen:
            uniq_files.append(f)
            seen.add(f)

    copied = 0
    skipped = 0
    for src in src_dirs:
        under_src = [f for f in uniq_files if str(f).startswith(str(src))]
        c, s = copy_files(under_src, dest_dir, relative_to=src, overwrite=args.overwrite, dry_run=args.dry_run)
        copied += c
        skipped += s

    print()
    print(f"Done. Copied: {copied}, Skipped: {skipped}, Total discovered: {len(uniq_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
