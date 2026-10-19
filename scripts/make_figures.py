"""Plot metrics where the starter supplies no plot; copy Cartucho's mAP plot."""
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
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
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), gridspec_kw={'width_ratios': [1, 2]})
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
            axes[1].set(xlabel='Epoch', xlim=(5, 50),
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


def q2_histograms():
    events = EventAccumulator(str(ROOT.parent / 'runs/q2_epoch_revision/q2_epoch_rerun'),
                              size_guidance={'histograms': 0, 'scalars': 0}).Reload()
    rates = events.Scalars('learning_rate')
    steps_per_epoch = rates[0].step
    if len(rates) != 50 or rates[-1].step != 50 * steps_per_epoch:
        raise RuntimeError('Finish all 50 Q2 epochs before generating histogram panels.')
    layers = [('resnet.layer1.1.conv1.weight/grad', 'q2_hist_conv.png'),
              ('resnet.layer4.0.bn2.bias/grad', 'q2_hist_bn.png')]
    selected = [[min(events.Histograms(tag), key=lambda v: abs((v.step + 1) / steps_per_epoch - epoch))
                 for epoch in [1, 25, 50]] for tag, filename in layers]
    # All summaries use the same native TensorBoard bucket grid. Select
    # common boundaries and combine complete buckets, without interpolation.
    grid = np.unique(np.concatenate([v.histogram_value.bucket_limit for row in selected for v in row]))
    indices = np.searchsorted(grid, np.linspace(grid[0], grid[-1], 41)).clip(0, len(grid) - 1)
    edges = np.unique(grid[indices])
    densities = []
    for row in selected:
        layer_densities = []
        for value in row:
            h = value.histogram_value
            assert h.bucket[0] == 0
            counts = np.zeros(len(edges) - 1)
            for low, high, count in zip(h.bucket_limit[:-1], h.bucket_limit[1:], h.bucket[1:]):
                if count:
                    index = np.searchsorted(edges, high, side='left') - 1
                    assert edges[index] <= low and high <= edges[index + 1]
                    counts[index] += count
            assert counts.sum() == h.num
            density = counts / (h.num * np.diff(edges))
            assert np.isclose(np.sum(density * np.diff(edges)), 1)
            layer_densities.append(density)
        densities.append(layer_densities)
    ymax = max(density.max() for row in densities for density in row) * 1.08
    limit = max(abs(edges[0]), abs(edges[-1]))
    np.savez_compressed(ROOT.parent / 'runs/q2_epoch_revision/histogram_panels.npz',
                        edges=edges, densities=np.array(densities),
                        steps=np.array([[v.step for v in row] for row in selected]),
                        parameter_counts=np.array([row[0].histogram_value.num for row in selected]))
    for (tag, filename), row, layer_densities in zip(layers, selected, densities):
        fig, axes = plt.subplots(1, 3, figsize=(8, 2.8), sharex=True, sharey=True)
        for ax, value, density in zip(axes, row, layer_densities):
            ax.stairs(density, edges, fill=True, alpha=.65)
            ax.axvline(0, color='black', linewidth=.6, linestyle=':')
            ax.set(title=f'Epoch {(value.step + 1) / steps_per_epoch:.2f}',
                   xlabel='Gradient', xlim=(-limit, limit), ylim=(0, ymax))
            ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
            ax.locator_params(axis='x', nbins=3)
        axes[0].set_ylabel('Probability density')
        fig.suptitle(tag.removeprefix('resnet.').removesuffix('/grad') +
                     f' ({int(row[0].histogram_value.num):,} parameters)')
        fig.tight_layout()
        fig.savefig(FIGURES / filename, dpi=180)
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
    for mode in ['overfit', 'full']:
        components = [scalars('fcos_' + mode, 'train/loss_' + kind + '_' + mode)
                      for kind in ['cls', 'box', 'ctr']]
        fig, ax = plt.subplots(figsize=(7, 3.2))
        steps = [v.step for v in components[0]]
        assert all([v.step for v in values] == steps for values in components)
        ax.plot(steps, [sum(v.value for v in row) for row in zip(*components)], linewidth=.7)
        ax.set(xlabel='Iteration', ylabel='Total training loss')
        fig.tight_layout()
        fig.savefig(FIGURES / f'q3_{mode}.png', dpi=180)
        plt.close(fig)
    # This image is produced by the unmodified evaluator's draw_plot_func.
    shutil.copy2(ROOT.parent / 'tools/mAP/output/mAP.png', FIGURES / 'q3_map.png')
    print('Selected Q1:', best)


if __name__ == '__main__':
    main()
