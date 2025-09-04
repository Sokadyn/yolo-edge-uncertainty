# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.results import ResultsUncertainty
from ultralytics.utils import ops
from ultralytics.models.yolo.detect import DetectionPredictor

class DetectionPredictorUncertainty(DetectionPredictor):
    """
    A class extending the YOLO DetectionPredictor class for handling additional uncertainty values.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """
        Post-process predictions and return a list of Results objects.
        """
        save_feats = getattr(self, "_feats", None) is not None
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names), # important, set number of classes so ops.non_max_suppression can handle it in extra information fields
            end2end=getattr(self.model, "end2end", False),
            rotated=self.args.task == "obb",
            return_idxs=save_feats,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        if save_feats:
            obj_feats = self.get_obj_feats(self._feats, preds[1])
            preds = preds[0]

        results = self.construct_results(preds, img, orig_imgs, **kwargs)

        if save_feats:
            for r, f in zip(results, obj_feats):
                r.feats = f  # add object features to results

        return results


    def construct_result(self, pred, img, orig_img, img_path):
        """
        Construct a single Results object from one image prediction with additional uncertainty values.
        """
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        return ResultsUncertainty(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :7])
