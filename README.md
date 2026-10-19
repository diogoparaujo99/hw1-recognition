# hw1

We strongly suggest using conda (or miniconda) for package management. We tested this assignment using a conda environment with python 3.10.

```shell
conda env create -f environment.yml
conda activate cmu16824hw1
```

`requirements.txt` is unchanged from the starter. `constraints.txt` pins compatible dependencies: in particular, Matplotlib 3.5.3 supports the original `canvas.set_window_title` and `canvas.tostring_rgb` calls. The environment disables user-site packages so they cannot override its versions. It uses the Agg backend for this machine's headless plotting.

**Note:** If you are using a Blackwell GPU (e.g. RTX 5080 or 5090), you need a newer PyTorch version (2.7+) and CUDA 12.8+.

## Data and outputs

Run classification commands from the repository root and detection commands from `detection/`, as in the starter. Downloaded data, checkpoints, and TensorBoard logs stay outside the repository:

```text
../data/                       VOC2007 images, XML and detection JSON
../checkpoints/starter_revision/ Q2 and detector checkpoints
../checkpoints/q1_bce_revision/  Fresh Q1 checkpoints
../runs/starter_revision/       Q2 and detector events and terminal logs
../runs/q1_bce_revision/         Fresh Q1 events, commands and measurements
../tools/mAP/                   Unmodified Cartucho evaluator
../build/                       Compiled report and temporary files
../submission/                  PDF and Python files only
```

```shell
bash q1_q2_classification/download_data.sh
cp data/voc07_*.json ../data/
export VLR_LOG_ROOT="$PWD/../runs/starter_revision"
export VLR_CHECKPOINT_ROOT="$PWD/../checkpoints/starter_revision"
```

The download script retains the starter's download/extract workflow but uses a mirror because its Internet Archive URLs return HTML. The VOC2007 archive MD5 checksums are `c52e279531787c972589f7e41ab4ae64` (trainval) and `b6e924de25625d8de591ea690078ad9f` (test).

## Q1 and Q2

Hyperparameters are defined directly in the original `ARGS` blocks. The original training loop, evaluation interval, normalization, DataLoader settings, TensorBoard tags, gradient logging, and checkpoint serialization are retained.

```shell
export VLR_LOG_ROOT="$PWD/../runs/q1_bce_revision"
export VLR_CHECKPOINT_ROOT="$PWD/../checkpoints/q1_bce_revision"
VLR_RUN_NAME=q1_bce_lr001_noaug VLR_AUGMENT=0 VLR_LR=0.001 python q1_q2_classification/train_q1.py
VLR_RUN_NAME=q1_bce_lr001_aug VLR_AUGMENT=1 VLR_LR=0.001 python q1_q2_classification/train_q1.py
VLR_RUN_NAME=q1_bce_lr0003_noaug VLR_AUGMENT=0 VLR_LR=0.0003 python q1_q2_classification/train_q1.py
VLR_RUN_NAME=q1_bce_lr0003_aug VLR_AUGMENT=1 VLR_LR=0.0003 python q1_q2_classification/train_q1.py
export VLR_LOG_ROOT="$PWD/../runs/starter_revision"
export VLR_CHECKPOINT_ROOT="$PWD/../checkpoints/starter_revision"
VLR_RUN_NAME=q2 python q1_q2_classification/train_q2.py
```

The only extra experiment controls are environment variables for the required augmentation comparison, learning-rate comparison, run names, and output directories. Q1 runs five epochs at 64×64; Q2 runs 50 epochs at 224×224. Both use seed 0 and the supplied architecture/optimizer choices. Q1 uses `BCEWithLogitsLoss(reduction="none")` directly on logits, then averages over the difficult-label mask with `(elementwise_loss * wgt).sum() / wgt.sum().clamp_min(1)`. The assignment permits a PyTorch loss; this explicitly authorized choice overrides the loss TODO comment for Q1. A model-type check keeps Q2 on its existing manual stable BCE. Ordinary inputs, extreme logits, masked labels, and all-masked batches were checked for matching losses and gradients; Q2 was checked separately. Q2 replaces the pretrained classifier with a 512-to-20 linear layer and fine-tunes all layers.

Fresh Q1 commands, settings, source hashes, loss checks, and measured results are recorded in `../runs/q1_bce_revision/`. The original tags are `Loss/train`, `map`, and `learning_rate`. Both protected evaluation functions remain unchanged. Classification mAP therefore averages AP computed separately in test batches of 1,000. Final evaluation explicitly uses `model.eval()`.

## Q3

Follow the original [detection README](detection/README.md):

```shell
cd detection
python -m test_object_detection
python train.py --visualize_gt
python train.py --overfit
python train.py
# Clone once, if it is not already present:
git clone https://github.com/Cartucho/mAP.git ../../tools/mAP
python train.py --inference --test_inference
python train.py --inference
cd ..
```

The provided 250-iteration overfit and 9,000-iteration full-training settings are unchanged. The three-level FPN, prediction heads, target transforms, losses, and inference fill the scaffold's TODOs. The supplied validation center crop and ten-image inference grid are preserved. The Cartucho checkout used here is `3605865a350859e60c7b711838d09c4e0012c774`.

Figure 3.3 is copied directly from `../tools/mAP/output/mAP.png`, produced by Cartucho's original `draw_plot_func`. No custom AP plot or replacement metric is used. Its plotting code and the starter's visualization helper are unmodified.

The few correctness fixes outside TODOs are moving the detector writer's close call after training, retaining ground truth for images with zero predictions, honoring the evaluation output directory, and putting the classifier in evaluation mode for its final score. The final score is logged before closing the classification writer; the `SimpleCNN` import selects the Q1 loss without changing function signatures. Other changes outside TODOs keep large artifacts out of the source/submission folders and set the detector's seed to 0.

## Report and submission

```shell
python scripts/tsne_features.py
python scripts/make_figures.py
python -m tensorboard.main --logdir ../runs/starter_revision --host 127.0.0.1 --port 6006 --load_fast=false
```

For the fresh Q1 dashboard, use `--logdir ../runs/q1_bce_revision` in the TensorBoard command.

The assignment provides no t-SNE implementation or combined scalar/loss plotting function, so the small scripts handle those deliverables. t-SNE uses the final 50-epoch model, 1,000 distinct test images, seed 0, the 512-dimensional pooled features, sklearn, and the required mean RGB class colors. Detector loss curves sum the three losses already logged by the starter.

For screenshots, install optional browser tooling with `python -m pip install playwright` and `python -m playwright install chromium`, then run:

```shell
python scripts/capture_tensorboard.py --question q2
python scripts/capture_tensorboard.py --question q3
```

These capture TensorBoard's own histogram rendering and the visible first two images of the unchanged ten-image grid. The original LaTeX packages, questions, and collaboration survey are preserved. Complete the survey before submitting.

Compile the top-level `main.tex` in Overleaf, or locally:

```shell
../tools/tectonic main.tex -o ../build
python scripts/package_submission.py
```

The packaging script puts only the report PDF and Python files in `../submission/` and `../submission.zip`; both must be under 100 MB. See [OVERLEAF.md](OVERLEAF.md) for synchronization.
