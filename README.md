# Increasing Trustworthiness of Edge AI by Adding Uncertainty Estimation to Object Detection

This repository contains the experiments for the corresponding [Diploma Thesis](https://repositum.tuwien.at/handle/20.500.12708/221805).

## YOLO Edge Uncertainty Framework (Ultralytics Fork)

> Fork of [Ultralytics YOLO](README_Original.md) with additional uncertainty estimation for object detection.

A comprehensive extension of Ultralytics YOLO that adds real-time uncertainty estimation capabilities for edge deployment scenarios. This framework provides multiple uncertainty estimation methods while focusing on minimizing computational overhead, making it suitable for resource-constrained environments. 
The framework is an experimental state, not ready for any production use.

## Environment

Use Python 3.10.12 and a virtual environment. Install the package and dependencies as follows:

```
python3 -m venv env
source env/bin/activate
pip install --upgrade pip wheel setuptools
pip install -e .
pip install -r requirements.txt
```

Hardware used for our experiments:
- AMD EPYC 9354P 32-Core Processor
- 500 GB RAM
- NVIDIA A100 GPU

## 📋 Methods Used

### 1. **Base Confidence**
- **Method**: `1 - max(conf)` per detection
- **Overhead**: Minimal
- **Config**: `yolo11n-base-confidence.yaml`
- **Eval Uncertainty**: `sigmoid-complement`

### 2. **Ensemble**
- **Method**: Multiple heads / predictions
- **Overhead**: Medium
- **Config**: `yolo11n-ensemble.yaml`
- **Eval Uncertainty**: `softmax-entropy`

### 3. **MC Dropout**
- **Method**: Stochastic forward passes with dropout
- **Overhead**: Medium
- **Config**: `yolo11n-mc-dropout.yaml`
- **Eval Uncertainty**: `softmax-entropy`

### 4. **EDL MEH**
- **Method**: Evidential Deep Learning with Model Evidence Head
- **Overhead**: Low
- **Config**: `yolo11n-edl-meh.yaml`
- **Eval Uncertainty**: `vacuity`

## Experiments
Run experiments in the following order (matching `yolo_edge_uncertainty/scripts`):

- Prepare datasets: use scripts in `dataset_gens` to create Cityscapes/KITTI train splits and validation splits (Foggy, RainCityscapes, BDD100K, nuImages).
- 1) Hyperparameter tuning: `python yolo_edge_uncertainty/scripts/0_hyperparameter_tuning.py`
- 2) Analyze tuning (notebook): open `yolo_edge_uncertainty/scripts/1_hyperparameter_tuning_analysis.ipynb`
- 3) Train + validate across splits: `python yolo_edge_uncertainty/scripts/2_generate_results.py`
- 4) Analyze results (notebook): open `yolo_edge_uncertainty/scripts/3_analyze_results.ipynb`
- 5) Collect best weights into repo: `python yolo_edge_uncertainty/scripts/4_copy_models_from_results.py`
- 6) Benchmark inference time: `python yolo_edge_uncertainty/scripts/5_benchmarking_inference_time.py --device cpu` (or `--device 0` for GPU)
- 7) Analyze inference speed (notebook): open `yolo_edge_uncertainty/scripts/6_inference_time_analysis.ipynb`
- Optional: export all best weights to ONNX: `python yolo_edge_uncertainty/scripts/export_best_to_onnx.py`
- Optional: run the example notebook: `yolo_edge_uncertainty/scripts/inference_examples.ipynb`

Outputs and artifacts:
- Training/validation results: `results/detect/data_splits_and_models/...`
- Copied best weights: `yolo_edge_uncertainty/models/.../weights/best.pt`
- Benchmark CSVs: `yolo_edge_uncertainty/csv/benchmark_inference_time_*.csv`

Metrics and experimental setup may be refined over time.

## 📄 License

This project is licensed under the AGPL-3.0 License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on top of [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- Our lab team
- Community contributions and feedback

---

**Note**: This framework is designed for research. Always validate uncertainty estimates in your specific application domain. 
