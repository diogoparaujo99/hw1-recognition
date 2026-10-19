# VLR Homework 1: classification and detection

The supplied Python modules implement VOC2007 multi-label classification and the assignment's FCOS detector. `report/` contains the provided LaTeX template and measured figures. The dataset, weights, logs, build files, and submission live outside this repository.

## Measured results

| Experiment | Training budget | Assignment score |
| --- | --- | --- |
| Q1 CNN, augmentation, Adam 0.001 | 5 epochs | 0.2674 mAP |
| Q2 ResNet-18 | 50 epochs | 0.8348 final mAP; 0.8365 best at epoch 6 |
| Q3 FCOS | 9,000 iterations | 36.90% mAP at IoU 0.5 |

These are single-seed results using the supplied evaluation procedures. The best Q2 checkpoint is used for t-SNE. All eight supplied detector tests and five focused additional checks pass.

## Setup

Run commands from this repository root. The defaults assume the repository is at `/home/daraujo/courses/vlr/hw1-recognition`.

```bash
uv venv --python 3.10 ../.venv
source ../.venv/bin/activate
uv pip install -r requirements.txt
python scripts/prepare_data.py
```

The original Internet Archive links returned HTML instead of archives. The download script uses a mirror and verifies the original VOC MD5 checksums before extraction. It checks all 5,011 trainval and 4,952 test images and annotations. The small detector JSON files supplied in `data/` are copied to the external data directory.

## Classification

```bash
python q1_q2_classification/train_q1.py --run-name q1_lr001_noaug --no-augment
python q1_q2_classification/train_q1.py --run-name q1_lr001_aug
python q1_q2_classification/train_q1.py --lr 0.0003 --run-name q1_lr0003_noaug --no-augment
python q1_q2_classification/train_q1.py --lr 0.0003 --run-name q1_lr0003_aug
python q1_q2_classification/train_q2.py --run-name q2_lr0001
```

Q1 uses 64-pixel inputs, Adam, batch size 64, five epochs, and StepLR decay by 0.5 after epoch three. Mild padded crops and horizontal flips are compared with deterministic preprocessing. The convolution/pooling output is 64 × 16 × 16, so the flattened dimension is 16,384.

Q2 uses 224-pixel inputs as required by the starter, ImageNet ResNet-18 weights, and a new 512-to-20 linear layer. All layers are fine-tuned with Adam at 0.0001 for 50 epochs. StepLR multiplies the rate by 0.3 every 15 epochs.

Both classifiers use independent binary labels and a stable manual binary cross-entropy. If any object of a class is marked difficult, that image/class loss has zero weight, as the starter specifies. Evaluation always uses deterministic transforms and `model.eval()`. The protected `compute_ap` and `eval_dataset_map` functions are unchanged. The latter averages AP computed separately in batches of 1,000 images; reported assignment mAP uses this exact procedure. Test-set curves and checkpoint selection follow the assignment; they should not be interpreted as an untouched external evaluation.

Each run writes `config.json`, `history.json`, and TensorBoard events to `../runs/<run-name>/`. Best and last weights are in `../checkpoints/<run-name>/`. "Best" means the highest observed assignment test mAP. Path overrides are available through `--data-root`, `--log-root`, and `--checkpoint-root`.

## FCOS

```bash
(cd detection && python -m test_object_detection)
python scripts/check_implementation.py
python detection/train.py --visualize_gt
python detection/train.py --overfit
python detection/train.py

git clone https://github.com/Cartucho/mAP.git ../tools/mAP
python detection/train.py --inference --test_inference
python detection/train.py --inference
```

The detector follows the supplied scaffold: pretrained RegNet-X-400MF, three FPN levels with strides 8/16/32 and 64 channels, and separate classification and box stems of two convolutions each. The box and centerness heads share the box stem. Supervision uses the supplied location-to-box matching, sigmoid focal classification loss, quarter-scaled L1 box loss, and centerness BCE. Losses use the supplied EMA foreground normalizer. Inference combines class probability and centerness by geometric mean and applies class-specific NMS.

The overfit experiment uses 160 fixed training images, 250 iterations, and SGD at 0.005. Full training uses 9,000 iterations and SGD at 0.008, batch size 16, momentum 0.9, and weight decay 0.0001. Learning rates decay at 60% and 90% of the iterations. The supplied tiny detection splits contain 2,501 training and 2,510 validation images. The supplied center-crop preprocessing is preserved, including its use on validation images; scores describe that transformed dataset.

Training logs are in `../runs/fcos_overfit/` and `../runs/fcos_full/`. Weights are in `../checkpoints/fcos_overfit.pt` and `../checkpoints/fcos_full.pt`. Evaluation uses confidence 0.4, NMS IoU 0.6, and evaluator IoU 0.5. Empty predictions produce an empty detection file and a populated ground-truth file, so missed objects count. The evaluator and its inputs/outputs stay in `../tools/mAP/`.

## Report and submission

```bash
python scripts/tsne_features.py
python scripts/make_figures.py
tensorboard --logdir ../runs --host 127.0.0.1 --port 6006 --load_fast=false
# In another terminal, with the same environment:
python scripts/capture_tensorboard.py
python scripts/write_report.py
```

Run report generation only after both full training runs and evaluation finish. The script checks completion and uses the recorded results.

The t-SNE uses 1,000 distinct test images, random seed 0, the selected fine-tuned model's 512-dimensional pooled features, perplexity 30, PCA initialization, automatic learning rate, and 1,500 iterations. Multi-label colors are arithmetic means of active GT class RGB colors; the legend lists the 20 base classes.

For screenshot capture, install the optional browser tooling with `uv pip install playwright` and `playwright install chromium`. It captures the actual TensorBoard dashboard. Other plots are rendered from TensorBoard events or recorded detector losses and evaluator output.

Compile `main.tex` with the provided template packages using Overleaf or a local LaTeX installation. Local commands used here:

```bash
../tools/tectonic main.tex -o ../build
python scripts/package_submission.py
python scripts/validate_artifacts.py
```

The packaging script copies only the completed PDF and required `.py` sources into `../submission/`, creates `../submission.zip`, and checks both unpacked and ZIP sizes against 100,000,000 bytes. Dataset files, weights, events, and third-party checkouts are excluded. The collaboration survey is left for the student to complete before the final submission.

See [OVERLEAF.md](OVERLEAF.md) for GitHub synchronization instructions. `environment-lock.txt` records the exact installed environment, including optional report-validation tools.
