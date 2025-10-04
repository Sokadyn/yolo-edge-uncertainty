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
                class_name = head.__class__.__name__
                if hasattr(head, 'num_mc_forward_passes'):
                    head.num_mc_forward_passes = int(train_args.get('num_mc_forward_passes'))
                if hasattr(head, 'num_ensemble_heads'):
                    head.num_ensemble_heads = int(train_args.get('num_ensemble_heads'))
                if hasattr(head, 'set_meh_lambda_activation_idx'):
                    meh_lambda = int(train_args.get('meh_lambda_activation_idx'))
                    head.set_meh_lambda_activation_idx(meh_lambda)
                if hasattr(head, 'num_dirichlet_samples'):
                    head.num_dirichlet_samples = int(train_args.get('num_dirichlet_samples'))
                if hasattr(head, 'set_dropout_rates'):
                    if class_name == 'DetectMCDropout':
                        dropout_rate = train_args.get('mc_dropout_rate')
                        dropout_method_idx = int(train_args.get('mc_dropout_method_idx'))
                        dropblock_size = int(train_args.get('mc_dropblock_size'))
                    elif class_name == 'DetectEnsemble':
                        dropout_rate = train_args.get('ensemble_dropout_rate')
                        dropout_method_idx = int(train_args.get('ensemble_dropout_method_idx'))
                        dropblock_size = int(train_args.get('ensemble_dropblock_size'))
                    head.set_dropout_rates(dropout_rate, dropout_method_idx, dropblock_size)
                    if verbose and RANK == -1:
                        LOGGER.info(
                            f"Updated model dropout ({class_name}): rate={dropout_rate}, method_idx={dropout_method_idx}, dropblock_size={dropblock_size}")
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

    def _freeze_batchnorm(self):
        """Put all BatchNorm2d modules into eval mode and freeze affine params (always on)."""
        model = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
        bn_layers = 0
        frozen_params = 0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                bn_layers += 1
                m.eval()
                for p in m.parameters():
                    if p.requires_grad:
                        p.requires_grad = False
                        frozen_params += p.numel()
        if RANK in {-1, 0}:
            msg = f"Freeze BN: set {bn_layers} BatchNorm2d layers to eval, froze {frozen_params} params"
            LOGGER.info(msg)

    def _model_train(self):
        super()._model_train()
        if getattr(self.args, 'freeze_bn', False):
            self._freeze_batchnorm()

    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        if getattr(self.args, 'freeze_bn', False):
            self._freeze_batchnorm()
