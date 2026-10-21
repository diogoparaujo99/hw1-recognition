"""Plot metrics where the starter supplies no plot; copy Cartucho's mAP plot."""
from pathlib import Path
import json
import shutil
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT.parent / 'runs' / 'starter_revision'
FIGURES = ROOT / 'report' / 'figures'


def scalars(run, tag):
    root = ROOT.parent / 'runs/q1_bce_revision' if run.startswith('q1_bce_') else RUNS
    if run == 'q2_epoch_rerun':
        root = ROOT.parent / 'runs/q2_epoch_revision'
    events = EventAccumulator(str(root / run), size_guidance={'scalars': 0}).Reload()
    return events.Scalars(tag)


def plot(runs, tag, filename, ylabel):
    zoom = filename == 'q2_map.png'
    if zoom:
        fig, axes = plt.subplots(2, 1, figsize=(7, 5.2))
        ax = axes[0]
    else:
        fig, ax = plt.subplots(figsize=(7, 3.2))
    q1 = filename.startswith('q1_')
    q2 = filename.startswith('q2_')
    for run, label in runs:
        values = scalars(run, tag)
        steps = [v.step for v in values]
        if q1 or q2:
            rates = scalars(run, 'learning_rate')
            steps_per_epoch = rates[0].step
            if q2 and (len(rates) != 50 or rates[-1].step != 50 * steps_per_epoch):
                raise RuntimeError('Finish all 50 Q2 epochs before generating report figures.')
            if q2 and tag == 'map' and values[-1].step != rates[-1].step:
                raise RuntimeError('Wait for the final Q2 evaluation before plotting mAP.')
            # Minibatch tags use zero-based indices; the final mAP and LR
            # use completed-update counts in the starter's logging loop.
            if tag != 'learning_rate':
                steps = [min(step + 1, rates[-1].step) for step in steps]
            steps = [step / steps_per_epoch for step in steps]
        if tag == 'learning_rate':
            ax.step([0] + steps,
                    [values[0].value] + [v.value for v in values], where='post', label=label)
        else:
            ax.plot(steps, [v.value for v in values], label=label)
        if zoom:
            axes[1].plot(steps, [v.value for v in values], marker='.', markersize=3)
            late = [v.value for step, v in zip(steps, values) if step >= 5]
            margin = max((max(late) - min(late)) * .1, .0005)
            axes[1].set(xlabel='Epoch', ylabel=ylabel, xlim=(5, 50),
                        ylim=(min(late) - margin, max(late) + margin), title='Epochs 5–50 (zoom)')
            ax.set_title('Full range')
    ax.set(xlabel='Epoch' if q1 or q2 else 'Training step', ylabel=ylabel)
    if q1 or q2:
        last_epoch = 5 if q1 else 50
        ax.set_xlim(0, last_epoch)
        ax.set_xticks(range(0, last_epoch + 1, 1 if q1 else 10))
    if filename == 'q2_loss.png':
        ax.set_yscale('log')
    if len(runs) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=180)
    plt.close(fig)


def histogram_density(histogram, edges):
    """Combine complete TensorBoard buckets and preserve their total count."""
    assert histogram.bucket[0] == 0
    counts = np.zeros(len(edges) - 1)
    buckets = zip(histogram.bucket_limit[:-1], histogram.bucket_limit[1:], histogram.bucket[1:])
    for low, high, count in buckets:
        if count:
            index = np.searchsorted(edges, high, side='left') - 1
            assert edges[index] <= low and high <= edges[index + 1]
            counts[index] += count
    assert counts.sum() == histogram.num
    density = counts / (histogram.num * np.diff(edges))
    assert np.isclose(np.sum(density * np.diff(edges)), 1)
    return density


def gradient_std(histogram):
    """Use the exact first and second moments stored in TensorBoard."""
    mean = histogram.sum / histogram.num
    variance = histogram.sum_squares / histogram.num - mean ** 2
    return np.sqrt(max(variance, 0))


def q2_histograms():
    """Compare selected histograms and gradient spread throughout training."""
    events = EventAccumulator(str(ROOT.parent / 'runs/q2_epoch_revision/q2_epoch_rerun'),
                              size_guidance={'histograms': 0, 'scalars': 0}).Reload()
    rates = events.Scalars('learning_rate')
    steps_per_epoch = rates[0].step
    if len(rates) != 50 or rates[-1].step != 50 * steps_per_epoch:
        raise RuntimeError('Finish all 50 Q2 epochs before generating histogram panels.')
    layers = [('resnet.layer1.1.conv1.weight/grad', 'q2_hist_conv.png'),
              ('resnet.layer4.0.bn2.bias/grad', 'q2_hist_bn.png')]
    histories = [events.Histograms(tag) for tag, _ in layers]
    selected = []
    for history in histories:
        selected.append([min(history, key=lambda v: abs((v.step + 1) / steps_per_epoch - epoch))
                         for epoch in [1, 25, 50]])
    # All summaries use the same native TensorBoard bucket grid. Select
    # common boundaries and combine complete buckets, without interpolation.
    grid = np.unique(np.concatenate([v.histogram_value.bucket_limit for row in selected for v in row]))
    indices = np.searchsorted(grid, np.linspace(grid[0], grid[-1], 81)).clip(0, len(grid) - 1)
    edges = np.unique(grid[indices])
    densities = [[histogram_density(value.histogram_value, edges) for value in row]
                 for row in selected]
    spread = np.array([[gradient_std(value.histogram_value) for value in history]
                       for history in histories])
    ymax = max(density.max() for row in densities for density in row) * 1.08
    limit = max(abs(edges[0]), abs(edges[-1]))
    np.savez_compressed(ROOT.parent / 'runs/q2_epoch_revision/histogram_panels.npz',
                        edges=edges, densities=np.array(densities),
                        steps=np.array([[v.step for v in row] for row in selected]),
                        parameter_counts=np.array([row[0].histogram_value.num for row in selected]),
                        history_steps=np.array([[v.step for v in history] for history in histories]),
                        history_std=spread)
    colors = ['#0072B2', '#D55E00', '#7B3294']
    linestyles = ['-', '--', ':']
    for layer_index, (tag, filename) in enumerate(layers):
        fig, axes = plt.subplots(2, 1, figsize=(7, 4.2), gridspec_kw={'height_ratios': [1.6, 1]})
        for value, density, color, style in zip(selected[layer_index], densities[layer_index], colors, linestyles):
            epoch = (value.step + 1) / steps_per_epoch
            axes[0].stairs(density, edges, color=color, linestyle=style, linewidth=1.5,
                           label=f'Epoch {epoch:.2f}')
            axes[1].scatter(epoch, gradient_std(value.histogram_value), color=color, s=25, zorder=3)
        axes[0].axvline(0, color='gray', linewidth=.6)
        axes[0].set(xlabel='Gradient', xlim=(-limit, limit), ylim=(0, ymax))
        axes[0].set_ylabel('Probability density')
        axes[0].ticklabel_format(axis='both', style='sci', scilimits=(0, 0), useMathText=True)
        axes[0].legend(frameon=False, fontsize=9)
        epochs = [(v.step + 1) / steps_per_epoch for v in histories[layer_index]]
        axes[1].plot(epochs, spread[layer_index], color='#334155', marker='.', markersize=3, linewidth=1)
        axes[1].set(xlabel='Epoch', ylabel='SD across parameters\n(log scale)', xlim=(0, 50),
                    ylim=(spread.min() * .85, spread.max() * 1.15), yscale='log')
        axes[1].yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        axes[1].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.0e}'))
        fig.suptitle(tag.removeprefix('resnet.').removesuffix('/grad') +
                     f' ({int(selected[layer_index][0].histogram_value.num):,} parameters)')
        fig.tight_layout()
        fig.savefig(FIGURES / filename, dpi=180)
        plt.close(fig)


def q3_loss_curves(overfit_run='fcos_160_val64_epoch400_seed0', overfit_output=None,
                   overfit_epochs=None):
    run_path = ROOT.parent / 'runs/q3_overfit_revision' / overfit_run
    history_path = run_path / 'history.json'
    for mode in ['overfit', 'full']:
        if mode == 'overfit' and history_path.exists():
            history = json.loads(history_path.read_text())
            settings = json.loads((run_path / 'settings.json').read_text())
            if history[-1]['epoch'] != settings['epochs']:
                raise RuntimeError('Finish the overfit diagnostic before updating the report.')
            if overfit_epochs is not None:
                history = [row for row in history if row['epoch'] <= overfit_epochs]
                if not history or history[-1]['epoch'] != overfit_epochs:
                    raise ValueError('The requested final epoch must have a recorded loss.')
            epochs = [row['epoch'] for row in history]
            best_epoch = min(history, key=lambda row: row['validation']['total'])['epoch']
            zoom_end = min(epochs[-1], max(150, ((best_epoch + 49) // 25) * 25))
            fig, axes = plt.subplots(2, 1, figsize=(7, 5.2))
            for ax in axes:
                ax.plot(epochs, [row['train']['total'] for row in history],
                        label=f"Training subset ({settings['train_images']} images)")
                ax.plot(epochs, [row['validation']['total'] for row in history],
                        label=f"Validation subset ({settings['validation_images']} images)", linestyle='--')
                ax.axvline(best_epoch, color='0.45', linestyle=':',
                           label=f'Lowest validation loss (epoch {best_epoch})')
                ax.set(xlabel='Epoch', ylabel='Total loss (fixed normalization)')
            title = 'Full run' if epochs[-1] == settings['epochs'] else f'Epochs 0–{epochs[-1]}'
            axes[0].set(xlim=(0, epochs[-1]), title=title)
            axes[0].legend()
            axes[1].set(xlim=(0, zoom_end), title=f'Early run (epochs 0–{zoom_end})')
            fig.tight_layout()
            fig.savefig(overfit_output or FIGURES / 'q3_overfit.png', dpi=180)
            plt.close(fig)
            continue
        components = [scalars('fcos_' + mode, 'train/loss_' + kind + '_' + mode)
                      for kind in ['cls', 'box', 'ctr']]
        fig, ax = plt.subplots(figsize=(7, 3.2))
        steps = [v.step for v in components[0]]
        assert all([v.step for v in values] == steps for values in components)
        # Starter batch size 16: 160/16=10 subset updates per epoch;
        # the full 2,501-image set has ceil(2501/16)=157 updates per epoch.
        updates_per_epoch = 10 if mode == 'overfit' else 157
        epochs = [(step + 1) / updates_per_epoch for step in steps]
        ax.plot(epochs, [sum(v.value for v in row) for row in zip(*components)], linewidth=.7)
        ax.set(xlabel='Epoch', ylabel='Total training loss', xlim=(0, epochs[-1]))
        fig.tight_layout()
        fig.savefig(FIGURES / f'q3_{mode}.png', dpi=180)
        plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    names = ['q1_bce_lr001_noaug', 'q1_bce_lr001_aug',
             'q1_bce_lr0003_noaug', 'q1_bce_lr0003_aug']
    best = max(names, key=lambda name: scalars(name, 'map')[-1].value)
    prefix = best.rsplit('_', 1)[0]
    comparison = [(prefix + '_noaug', 'Without augmentation'), (prefix + '_aug', 'With augmentation')]
    plot(comparison, 'Loss/train', 'q1_compare_loss.png', 'Training loss')
    plot(comparison, 'map', 'q1_compare_map.png', 'Test mAP')
    for run, question in [(best, 'q1'), ('q2_epoch_rerun', 'q2')]:
        for tag, suffix, label in [('Loss/train', 'loss', 'Training loss'),
                                   ('map', 'map', 'Test mAP'),
                                   ('learning_rate', 'lr', 'Learning rate')]:
            plot([(run, run)], tag, f'{question}_{suffix}.png', label)
    q2_histograms()
    q3_loss_curves()
    # This image is produced by the unmodified evaluator's draw_plot_func.
    shutil.copy2(ROOT.parent / 'tools/mAP/output/mAP.png', FIGURES / 'q3_map.png')
    print('Selected Q1:', best)


if __name__ == '__main__':
    main()
