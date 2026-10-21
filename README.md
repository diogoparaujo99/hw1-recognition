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
../checkpoints/starter_revision/ Detector and earlier Q2 checkpoints
../checkpoints/q1_bce_revision/  Fresh Q1 checkpoints
../checkpoints/q2_epoch_revision/ Fresh Q2 checkpoint
../runs/starter_revision/       Q2 and detector events and terminal logs
../runs/q1_bce_revision/         Fresh Q1 events, commands and measurements
../runs/q2_epoch_revision/       Fresh Q2 events, commands and measurements
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
export VLR_LOG_ROOT="$PWD/../runs/q2_epoch_revision"
export VLR_CHECKPOINT_ROOT="$PWD/../checkpoints/q2_epoch_revision"
VLR_RUN_NAME=q2_epoch_rerun VLR_LR=0.0001 VLR_AUGMENT=1 python q1_q2_classification/train_q2.py
```

The only extra experiment controls are environment variables for the required augmentation comparison, learning-rate comparison, run names, and output directories. Q1 runs five epochs at 64×64; Q2 runs 50 epochs at 224×224. Both use seed 0 and the supplied architecture/optimizer choices. Q1 uses `BCEWithLogitsLoss(reduction="none")` directly on logits, then averages over the difficult-label mask with `(elementwise_loss * wgt).sum() / wgt.sum().clamp_min(1)`. The assignment permits a PyTorch loss; this explicitly authorized choice overrides the loss TODO comment for Q1. A model-type check keeps Q2 on its existing manual stable BCE. Ordinary inputs, extreme logits, masked labels, and all-masked batches were checked for matching losses and gradients; Q2 was checked separately. Q2 replaces the pretrained classifier with a 512-to-20 linear layer and fine-tunes all layers.

Fresh Q1 commands, settings, source hashes, loss checks, and measured results are recorded in `../runs/q1_bce_revision/`. The original tags are `Loss/train`, `map`, and `learning_rate`. Both protected evaluation functions remain unchanged. Classification mAP therefore averages AP computed separately in test batches of 1,000. Final evaluation explicitly uses `model.eval()`.

## Q3

Follow the original [detection README](detection/README.md):

```shell
export VLR_LOG_ROOT="$PWD/../runs/starter_revision"
export VLR_CHECKPOINT_ROOT="$PWD/../checkpoints/starter_revision"
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

## Small-subset overfitting diagnostic

Figure 3.1 uses a separate diagnostic; the original 250-update run and the full-training checkpoint remain under `../runs/starter_revision/` and `../checkpoints/starter_revision/`. The diagnostic keeps the starter's 160 evenly spaced training images, 224-pixel inputs, batch size 16, seed 0, and SGD (learning rate 0.005, momentum 0.9, weight decay 0.0001). It extends training to 400 epochs, or 4,000 updates, with the same scheduler rule: multiply the rate by 0.1 at 60% and 90% of the updates.

Exactly 64 fixed, evenly spaced images from the separate validation split supply every validation measurement; they never enter the optimizer. This is 40% of the training count, the largest size in the requested 20–40% range. It provides four complete batches and includes all 20 classes after the supplied crop. The indices were fixed before examining the measured losses; paths and class counts are recorded with the run.

From `detection/`, run:

```shell
CUDA_VISIBLE_DEVICES=1 python overfit_diagnostic.py --epochs 400 --val-images 64 --run-name fcos_160_val64_epoch400_seed0
```

Logs, settings, source hashes, losses, and AP results are saved under `../runs/q3_overfit_revision/fcos_160_val64_epoch400_seed0/`; checkpoints go under `../checkpoints/q3_overfit_revision/fcos_160_val64_epoch400_seed0/`. Use a different run name for another experiment. Training and validation losses are measured at epochs 0, 1, 2, and every five epochs, in evaluation mode without gradients or changes to batch statistics. Both use the same fixed normalizer, 150, so changes in the training EMA cannot create an apparent validation-loss trend. The optimizer still uses the original EMA-normalized loss. Each plotted point averages all images in its subset.

The only changes to `FCOS.forward` allow evaluation with supplied ground-truth boxes to return losses and keep the EMA frozen during evaluation. Training and ordinary inference follow their existing paths. Runtime checks verify that each loss evaluation preserves the weights, buffers, and training normalizer.

At epochs 25, 100, and 400, the supplied inference writer and an unchanged copy of Cartucho's evaluator measure AP on both subsets (confidence 0.4, NMS IoU 0.6, evaluation IoU 0.5). These are diagnostic subset scores, separate from the full-validation result in Figure 3.3. Centerness has soft targets, so fitting those targets exactly does not require a zero total loss.

The lowest measured validation loss was 1.2392 at epoch 55, with an almost equal value of 1.2395 at epoch 115. The early curve fluctuates, so epoch 55 is not a precise onset of overfitting. Later, training loss stays low while validation loss rises. At epoch 400, training loss is 0.5488 and validation loss is 1.4364, 15.9% above its minimum.

| Epoch | Training-subset mAP | Validation-subset mAP |
| --- | --- | --- |
| 25 | 0.09% | 0.00% |
| 100 | 50.16% | 8.11% |
| 400 | 92.49% | 9.99% |

Both AP calculations include all 20 classes. The large final gap shows a strong fit to the small training subset with poor generalization; training detections are not perfect. Validation AP still improves across these checkpoints even though validation loss increases later, so the loss-based conclusion does not mean that every validation metric deteriorates.

To regenerate only the Q3 loss figures from the repository root:

```shell
python -c 'from scripts.make_figures import q3_loss_curves; q3_loss_curves()'
```

Views ending at epochs 150, 200, 300, and 400 are saved as `figure_3_1_epochs_*.png` in the run directory, alongside `epoch_comparison.json`. These show prefixes of the same 400-epoch run, with its scheduler unchanged. Validation loss rises 4.0%, 9.6%, 14.5%, and 15.9% above its minimum at those endpoints. Figure 3.1 uses the complete 400-epoch view, which shows the most sustained divergence, plus an early-training panel. No smoothing is applied. To reproduce a shorter view without changing the report figure:

```shell
python -c 'from scripts.make_figures import q3_loss_curves; q3_loss_curves(overfit_epochs=200, overfit_output="../runs/q3_overfit_revision/fcos_160_val64_epoch400_seed0/figure_3_1_epochs_200.png")'
```

The previous Figure 3.1, compiled PDF, and LaTeX source are preserved under `../runs/q3_overfit_revision/fcos_160_epoch400_seed0/saved_figure_3_1/`. That run used 320 validation images and had a 13.3% rise from its minimum validation loss. The new figure is a modestly clearer illustration with the requested 64 images; the different validation subsets do not establish an improvement in the model itself.

Both loss figures use epochs. The original full run has 157 updates per epoch (2,501 images, batch size 16); Figure 3.2 keeps its original loss values.

## Report and submission

```shell
python scripts/tsne_features.py
python scripts/make_figures.py
python -m tensorboard.main --logdir ../runs/starter_revision --host 127.0.0.1 --port 6006 --load_fast=false
```

Classification plots convert logged steps to epochs using the 79 minibatches per epoch. Q1 stops at five epochs and Q2 at 50. The Q2 mAP figure shows both the full raw curve and a labeled zoom of epochs 5–50; its loss figure uses a logarithmic y-axis. No smoothing or averaging is applied. Q2 retains loss logging every 100 updates and mAP evaluation every 70 updates.

For the fresh Q1 dashboard, use `--logdir ../runs/q1_bce_revision` in the TensorBoard command. For fresh Q2, use `--logdir ../runs/q2_epoch_revision --port 6007`. Its commands, unchanged training settings, source hashes, and results are recorded in that directory.

The assignment provides no t-SNE implementation or combined scalar/loss plotting function, so the small scripts handle those deliverables. t-SNE uses the final 50-epoch model, 1,000 distinct test images, seed 0, the 512-dimensional pooled features, sklearn, and the required mean RGB class colors. Detector loss curves sum the three losses already logged by the starter.

The starter already logs gradients using `SummaryWriter.add_histogram` after backpropagation and before the optimizer update. It provides no static histogram plotting template. Q2 overlays the histograms nearest epochs 1, 25, and 50 for each layer, labeled with the actual recorded epochs. Below each overlay, the exact gradient standard deviation across parameters is plotted for all 40 logged minibatches on a logarithmic scale. Colored points connect the selected histograms to the history.

Native TensorBoard buckets are combined onto common boundaries without interpolation. Counts are divided by parameter count and bin width so each density integrates to one. Both layers share gradient, density, and standard-deviation limits; no tails are excluded.

Spread is computed from the original logged first and second moments, independently of histogram binning. The complete history shows that the overall decrease is not monotonic. Plot data, including the full spread history, are saved outside the repository in `../runs/q2_epoch_revision/histogram_panels.npz`.

For optional native TensorBoard screenshots, install browser tooling with `python -m pip install playwright` and `python -m playwright install chromium`, then run:

```shell
python scripts/capture_tensorboard.py --question q2 --url http://127.0.0.1:6007
python scripts/capture_tensorboard.py --question q3
```

The Q2 command saves the native stacked display under `../runs/q2_epoch_revision/native_histograms/` for reference. The Q3 command takes the first six renderings from the starter's ten-image inference grid and arranges them in two columns. It saves this display under `../runs/starter_revision/fcos_inference_six/` and captures the TensorBoard image card. The original renderings, predictions, and ten-image log stay unchanged.

To take your own screenshot, open `http://localhost:6006/#images`, select only `fcos_inference_six`, and filter the tag to `test_images`. Enable **Show actual image size**. If VS Code is connected through Remote SSH, open the **Ports** view, forward port **6006**, and open its forwarded address in your browser.

The original LaTeX packages, questions, and collaboration survey are preserved. Complete the survey before submitting.

Compile the top-level `main.tex` in Overleaf, or locally:

```shell
../tools/tectonic main.tex -o ../build
python scripts/package_submission.py
```

The packaging script puts only the report PDF and Python files in `../submission/` and `../submission.zip`; both must be under 100 MB. See [OVERLEAF.md](OVERLEAF.md) for synchronization.
