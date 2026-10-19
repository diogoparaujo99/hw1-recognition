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
    for run, label in runs:
        values = scalars(run, tag)
        if tag == 'learning_rate':
            ax.step([0] + [v.step for v in values],
                    [values[0].value] + [v.value for v in values], where='post', label=label)
        else:
            ax.plot([v.step for v in values], [v.value for v in values], label=label)
    ax.set(xlabel='Training step', ylabel=ylabel)
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
