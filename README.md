# YOLO Edge Uncertainty Framework (Ultralytics Fork)

> Fork of [Ultralytics YOLO](README_Original.md) with additional uncertainty estimation for object detection.

A comprehensive extension of Ultralytics YOLO that adds real-time uncertainty estimation capabilities for edge deployment scenarios. This framework provides multiple uncertainty estimation methods while focusing on minimizing computational overhead, making it suitable for resource-constrained environments. 

Work is currently in progress, and the framework is an experimental state, not ready for any production use. Further tutorials and documentation will be added as well.

## 📋 Uncertainty Estimation Methods

### 1. **Base Confidence**
- **Method**: Uses `1 - max(confidence_scores)` as uncertainty
- **Overhead**: Minimal - no additional computation
- **Configuration**: `yolo11n-base-confidence.yaml`

### 2. **Base Uncertainty**
- **Method**: Calculates entropy of classification scores per detection
- **Overhead**: Low - entropy calculation only
- **Configuration**: `yolo11n-base-uncertainty.yaml`

### 3. **Ensemble**
- **Method**: Multiple model predictions for retrieving uncertainty
- **Overhead**: Medium - requires multiple samples, uses mutliple detection heads
- **Configuration**: `yolo11n-ensemble.yaml`

### 4. **MC Dropout**
- **Method**: Monte Carlo sampling with dropout during inference to estimate uncertainty
- **Overhead**: Medium - multiple forward passes with dropout, only in the detection head
- **Configuration**: `yolo11n-mc-dropout.yaml`

### 5. **EDL MEH**
- **Method**: Estimating uncertainty through Evidential Deep Learning (EDL) with Model Evidence Head (MEH), according to Park et al. 2023
- **Overhead**: Low - single forward pass, just one more output value per detection
- **Configuration**: `yolo11n-edl-meh.yaml`

## Experiments
To run experiments:
- Prepare the datasets (cityscapes, foggy-cityscpapes, raincityscapes and kitti) with scripts in `dataset_gens` (0-3)
- Run `yolo_edge_uncertainty_scripts/generate_results.py`
- Check results with `yolo_edge_uncertainty_scripts/analyse_results.ipynb`

Metrics and experimental setup will still be further refined in the future.

## 📄 License

This project is licensed under the AGPL-3.0 License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on top of [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- Our lab team
- Community contributions and feedback

---

**Note**: This framework is designed for research. Always validate uncertainty estimates in your specific application domain. 