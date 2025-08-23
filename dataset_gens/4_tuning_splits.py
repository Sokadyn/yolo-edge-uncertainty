"""
This script creates further train/val splits from the KITTI training data for efficient
hyperparameter tuning.

Usage:
    python3 4_tuning_splits.py

Output:
    Creates train.txt and val.txt files in the KITTI dataset directory
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ultralytics import settings
settings.update({"sync": False, "runs_dir": "interim_results"})
from ultralytics.data.split import autosplit

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
kitti_train_dir = os.path.join(parent_dir, 'datasets/kitti_from_coco80/images/train')

autosplit(
    kitti_train_dir,
    (0.8, 0.2, 0.0),
    annotated_only=True)