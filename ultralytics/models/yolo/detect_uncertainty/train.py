# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
import random
from copy import copy
from typing import Any, Dict, List, Optional

import numpy as np
import torch.nn as nn

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.nn.tasks import DetectionModelUncertainty
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
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
            train_args = None
            # If hyperparameters changed before training, update model accordingly
            if hasattr(weights, 'args'):
                train_args = weights.args.__dict__ if hasattr(weights.args, '__dict__') else weights.args
            if train_args:
                head = model.model[-1]
                if hasattr(head, 'num_mc_forward_passes'):
                    head.num_mc_forward_passes = train_args.get('num_mc_forward_passes', 10)
                if hasattr(head, 'num_ensemble_heads'):
                    head.num_ensemble_heads = train_args.get('num_ensemble_heads', 5)
                if hasattr(head, 'set_meh_lambda_activation_idx'):
                    meh_lambda = train_args.get('meh_lambda_activation_idx', 3)
                    head.set_meh_lambda_activation_idx(meh_lambda)
                if hasattr(head, 'set_dropout_rates'):
                    dropout_rate = train_args.get('dropout_rate', 0.05)
                    dropout_method_idx = train_args.get('dropout_method_idx', 0)
                    dropblock_size = train_args.get('dropblock_size', 3)
                    head.set_dropout_rates(dropout_rate, dropout_method_idx, dropblock_size)
                    if verbose and RANK == -1:
                        LOGGER.info(f"Updated model dropout: rate={dropout_rate}, method_idx={dropout_method_idx}, dropblock_size={dropblock_size}")
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
        return ("\n" + "%11s" * 9) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
            "mUE50",
            "mUE50-95"
        )

