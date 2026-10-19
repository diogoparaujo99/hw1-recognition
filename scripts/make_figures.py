"""Plot metrics where the starter supplies no plot; copy Cartucho's mAP plot."""
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT.parent / 'runs' / 'starter_revision'
FIGURES = ROOT / 'report' / 'figures'


def scalars(run, tag):
    root = ROOT.parent / 'runs/q1_bce_revision' if run.startswith('q1_bce_') else RUNS
    events = EventAccumulator(str(root / run), size_guidance={'scalars': 0}).Reload()
    return events.Scalars(tag)


def plot(runs, tag, filename, ylabel):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    q1 = filename.startswith('q1_')
    for run, label in runs:
        values = scalars(run, tag)
        steps = [v.step for v in values]
        if q1:
            rates = scalars(run, 'learning_rate')
            steps_per_epoch = rates[0].step
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
    ax.set(xlabel='Epoch' if q1 else 'Training step', ylabel=ylabel)
    if q1:
        ax.set_xlim(0, 5)
        ax.set_xticks(range(6))
    if len(runs) > 1:
        ax.legend()
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
    for run, question in [(best, 'q1'), ('q2', 'q2')]:
        for tag, suffix, label in [('Loss/train', 'loss', 'Training loss'),
                                   ('map', 'map', 'Test mAP'),
                                   ('learning_rate', 'lr', 'Learning rate')]:
            plot([(run, run)], tag, f'{question}_{suffix}.png', label)
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
