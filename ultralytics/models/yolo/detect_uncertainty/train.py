# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
import random
from copy import copy
from typing import Any, Dict, List, Optional

import numpy as np
import torch.nn as nn

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.nn.tasks import DetectionModelUncertainty
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first
from ultralytics.models.yolo.detect import DetectionTrainer
from .val import DetectionValidatorUncertainty

class DetectionTrainerUncertainty(DetectionTrainer):
    """
    A class extending the YOLO DetectionTrainer class for training detection models with uncertainty estimation.
    """

    def get_model(self, cfg: Optional[str] = None, weights: Optional[str] = None, verbose: bool = True):
        """
        Return a YOLO detection model.

        Args:
            cfg (str, optional): Path to model configuration file.
            weights (str, optional): Path to model weights.
            verbose (bool): Whether to display model information.

        Returns:
            (DetectionModel): YOLO detection model.
        """
        model = DetectionModelUncertainty(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return DetectionValidatorUncertainty(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def progress_string(self):
        """Return a formatted string of training progress with epoch, GPU memory, loss, instances, size, and uncertainty metrics."""
        return ("\n" + "%11s" * 11) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
            "mUE50",
            "mUE50-95",
            "max_mAP50_unc",
            "max_mAP50-95_unc",
        )

