# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, box_iou, DetMetricsUncertainty
from ultralytics.utils.plotting import plot_images_with_uncertainty
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.modules.head import initialize_uncertainty_layers


class DetectionValidatorUncertainty(DetectionValidator):
    """
    A class extending the YOLO DetectionValidator class for validation with additional uncertainty estimates.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """
        Initialize detection validator with necessary variables and settings.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to use for validation.
            save_dir (Path, optional): Directory to save results.
            args (Dict[str, Any], optional): Arguments for the validator.
            _callbacks (List[Any], optional): List of callback functions.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.metrics = DetMetricsUncertainty()
        self.uncertainty_bins = np.zeros(100)
        self.bin_edges = np.linspace(0, 10, 101)
        if hasattr(self, 'model') and self.model is not None:
            initialize_uncertainty_layers(self.model, self.args)

    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess batch of images for YOLO validation.

        Args:
            batch (Dict[str, Any]): Batch containing images and annotations.

        Returns:
            (Dict[str, Any]): Preprocessed batch.
        """
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        return batch

    def init_metrics(self, model: torch.nn.Module) -> None:
        """
        Initialize evaluation metrics for YOLO detection validation.

        Args:
            model (torch.nn.Module): Model to validate.
        """
        val = self.data.get(self.args.split, "")  # validation path
        self.is_coco = (
            isinstance(val, str)
            and "coco" in val
            and (val.endswith(f"{os.sep}val2017.txt") or val.endswith(f"{os.sep}test-dev2017.txt"))
        )  # is COCO
        self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco  # is LVIS
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(1, len(model.names) + 1))
        self.args.save_json |= self.args.val and (self.is_coco or self.is_lvis) and not self.training  # run final val
        self.names = model.names
        self.nc = len(model.names)
        self.end2end = getattr(model, "end2end", False)
        self.seen = 0
        self.jdict = []
        self.metrics.names = self.names
        self.confusion_matrix = ConfusionMatrix(names=list(model.names.values()))

    def get_desc(self) -> str:
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ("%22s" + "%11s" * 18) % (
            "Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95", "mUE50", "mUE50_t", "mUE50-95", "mUE50-95_t",
            "mUE50_c", "mUE50_i", "mUE50-95_c", "mUE50-95_i", "AUROC50", "AUROC50-95", 
            "max_mAP50", "max_mAP50-95"
        )

    def postprocess(self, preds: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """
        Apply Non-maximum suppression to prediction outputs.

        Args:
            preds (torch.Tensor): Raw predictions from the model.

        Returns:
            (List[Dict[str, torch.Tensor]]): Processed predictions after NMS, where each dict contains
                'bboxes', 'conf', 'cls', and 'unc' tensors.
        """
        outputs = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            nc=self.nc,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            end2end=self.end2end,
            rotated=self.args.task == "obb",
        )
        
        results = [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "unc": x[:, 6]} for x in outputs]

        for x in results:
            if x["unc"].numel() > 0:
                unc_values = x["unc"].cpu().numpy()
                bin_indices = np.digitize(unc_values, self.bin_edges[:-1])
                valid_indices = (bin_indices > 0) & (bin_indices <= len(self.uncertainty_bins))
                np.add.at(self.uncertainty_bins, bin_indices[valid_indices] - 1, 1)

        return results

    def plot_final_histogram(self):
        """Plot and save the histogram of accumulated uncertainty values."""
        plt.figure()

        bin_width = self.bin_edges[1] - self.bin_edges[0]
        plt.bar(self.bin_edges[:-1], self.uncertainty_bins, width=bin_width, color='blue', alpha=0.7, align='edge')

        head_name = self.args.model.replace('.pt', '').replace('.yaml', '') if hasattr(self.args, 'model') else 'UnknownHead'
        plt.title('Final Histogram of Uncertainty Values')
        plt.xlabel('Uncertainty Bins')
        plt.ylabel('Frequency')
        plt.xlim(self.bin_edges[0], self.bin_edges[-1])
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(str(self.save_dir / "uncertainty_histogram.png"))
        plt.close()


    def _prepare_pred(self, pred: Dict[str, torch.Tensor], pbatch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Prepare predictions for evaluation against ground truth.

        Args:
            pred (Dict[str, torch.Tensor]): Post-processed predictions from the model.
            pbatch (Dict[str, Any]): Prepared batch information.

        Returns:
            (Dict[str, torch.Tensor]): Prepared predictions in native space.
        """
        cls = pred["cls"]
        if self.args.single_cls:
            cls *= 0
        # predn = pred.clone()
        bboxes = ops.scale_boxes(
            pbatch["imgsz"], pred["bboxes"].clone(), pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )  # native-space pred
        return {"bboxes": bboxes, "conf": pred["conf"], "cls": cls, "unc": pred["unc"]}

    def update_metrics(self, preds: List[Dict[str, torch.Tensor]], batch: Dict[str, Any]) -> None:
        """
        Update metrics with new predictions and ground truth.

        Args:
            preds (List[Dict[str, torch.Tensor]]): List of predictions from the model.
            batch (Dict[str, Any]): Batch data containing ground truth.
        """
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred, pbatch)

            cls = pbatch["cls"].cpu().numpy()
            no_pred = len(predn["cls"]) == 0
            if no_pred and len(cls) == 0:
                continue
            self.metrics.update_stats(
                {
                    **self._process_batch(predn, pbatch),
                    "target_cls": cls,
                    "target_img": np.unique(cls),
                    "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                    "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                    "unc": np.zeros(0) if no_pred else predn["unc"].cpu().numpy(),
                }
            )
            # Evaluate
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, pbatch, conf=self.args.conf)

            if no_pred:
                continue

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(batch['im_file'][si]).stem}.txt",
                )

    def finalize_metrics(self) -> None:
        """Set final values for metrics speed and confusion matrix."""
        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(save_dir=self.save_dir, normalize=normalize, on_plot=self.on_plot)
                self.plot_final_histogram()
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def get_stats(self) -> Dict[str, Any]:
        """
        Calculate and return metrics statistics.

        Returns:
            (Dict[str, Any]): Dictionary containing metrics results.
        """
        self.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        self.metrics.clear_stats()
        return self.metrics.results_dict

    def print_results(self) -> None:
        """Print training/validation set metrics per class."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ("all", self.seen, self.metrics.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.metrics.nt_per_class.sum() == 0:
            LOGGER.warning(f"no labels found in {self.args.task} set, can not compute metrics without labels")

        # Print results per class
        pf = "%22s" + "%11i" * 2 + "%11.3g" * 4 # * 4 instead of len(self.metrics.keys) to skip uncertainty metrics (uncertainty not defined per class)
        if self.args.verbose and not self.training and self.nc > 1 and len(self.metrics.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(
                    pf
                    % (
                        self.names[c],
                        self.metrics.nt_per_image[c],
                        self.metrics.nt_per_class[c],
                        *self.metrics.class_result(i),
                    )
                )

    def _process_batch(self, preds: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Return correct prediction matrix.

        Args:
            preds (Dict[str, torch.Tensor]): Dictionary containing prediction data with 'bboxes' and 'cls' keys.
            batch (Dict[str, Any]): Batch dictionary containing ground truth data with 'bboxes' and 'cls' keys.

        Returns:
            (Dict[str, np.ndarray]): Dictionary containing 'tp' key with correct prediction matrix of shape (N, 10) for 10 IoU levels.
        """
        if len(batch["cls"]) == 0 or len(preds["cls"]) == 0:
            return {"tp": np.zeros((len(preds["cls"]), self.niou), dtype=bool)}
        iou = box_iou(batch["bboxes"], preds["bboxes"])
        tp = self.match_predictions(preds["cls"], batch["cls"], iou).cpu().numpy()
        return {"tp": tp}
    
    
    def plot_predictions(
        self, batch: Dict[str, Any], preds: List[Dict[str, torch.Tensor]], ni: int, max_det: Optional[int] = None
    ) -> None:
        """
        Plot predicted bounding boxes on input images and save the result.

        Args:
            batch (Dict[str, Any]): Batch containing images and annotations.
            preds (List[Dict[str, torch.Tensor]]): List of predictions from the model.
            ni (int): Batch index.
            max_det (Optional[int]): Maximum number of detections to plot.
        """
        # TODO: optimize this
        for i, pred in enumerate(preds):
            pred["batch_idx"] = torch.ones_like(pred["conf"]) * i  # add batch index to predictions
        keys = preds[0].keys()
        max_det = max_det or self.args.max_det
        batched_preds = {k: torch.cat([x[k][:max_det] for x in preds], dim=0) for k in keys}
        # TODO: fix this
        batched_preds["bboxes"][:, :4] = ops.xyxy2xywh(batched_preds["bboxes"][:, :4])  # convert to xywh format
        plot_images_with_uncertainty(
            images=batch["img"],
            labels=batched_preds,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred

    def save_one_txt(self, predn: Dict[str, torch.Tensor], save_conf: bool, shape: Tuple[int, int], file: Path) -> None:
        """
        Save YOLO detections to a txt file in normalized coordinates in a specific format.

        Args:
            predn (Dict[str, torch.Tensor]): Dictionary containing predictions with keys 'bboxes', 'conf', and 'cls'.
            save_conf (bool): Whether to save confidence scores.
            shape (Tuple[int, int]): Shape of the original image (height, width).
            file (Path): File path to save the detections.
        """
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn: Dict[str, torch.Tensor], filename: str) -> None:
        """
        Serialize YOLO predictions to COCO json format.

        Args:
            predn (Dict[str, torch.Tensor]): Predictions dictionary containing 'bboxes', 'conf', and 'cls' keys
                with bounding box coordinates, confidence scores, and class predictions.
            filename (str): Image filename.
        """
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn["bboxes"])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for b, s, c in zip(box.tolist(), predn["conf"].tolist(), predn["cls"].tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(c)],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(s, 5),
                }
            )

    def eval_json(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate YOLO output in JSON format and return performance statistics.

        Args:
            stats (Dict[str, Any]): Current statistics dictionary.

        Returns:
            (Dict[str, Any]): Updated statistics dictionary with COCO/LVIS evaluation results.
        """
        if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
            pred_json = self.save_dir / "predictions.json"  # predictions
            anno_json = (
                self.data["path"]
                / "annotations"
                / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
            )  # annotations
            pkg = "pycocotools" if self.is_coco else "lvis"
            LOGGER.info(f"\nEvaluating {pkg} mAP using {pred_json} and {anno_json}...")
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                for x in pred_json, anno_json:
                    assert x.is_file(), f"{x} file not found"
                check_requirements("pycocotools>=2.0.6" if self.is_coco else "lvis>=0.5.3")
                if self.is_coco:
                    from pycocotools.coco import COCO  # noqa
                    from pycocotools.cocoeval import COCOeval  # noqa

                    anno = COCO(str(anno_json))  # init annotations api
                    pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
                    val = COCOeval(anno, pred, "bbox")
                else:
                    from lvis import LVIS, LVISEval

                    anno = LVIS(str(anno_json))  # init annotations api
                    pred = anno._load_json(str(pred_json))  # init predictions api (must pass string, not Path)
                    val = LVISEval(anno, pred, "bbox")
                val.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # images to eval
                val.evaluate()
                val.accumulate()
                val.summarize()
                if self.is_lvis:
                    val.print_results()  # explicitly call print_results
                # update mAP50-95 and mAP50
                stats[self.metrics.keys[-1]], stats[self.metrics.keys[-2]] = (
                    val.stats[:2] if self.is_coco else [val.results["AP"], val.results["AP50"]]
                )
                if self.is_lvis:
                    stats["metrics/APr(B)"] = val.results["APr"]
                    stats["metrics/APc(B)"] = val.results["APc"]
                    stats["metrics/APf(B)"] = val.results["APf"]
                    stats["fitness"] = val.results["AP"]
            except Exception as e:
                LOGGER.warning(f"{pkg} unable to run: {e}")
        return stats
