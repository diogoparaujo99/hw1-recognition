"""Command-line settings shared by the two classification experiments."""
import argparse
from pathlib import Path
from utils import ARGS

COURSE_DIR = Path(__file__).resolve().parents[2]


def parse_args(question):
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3 if question == 1 else 1e-4)
    parser.add_argument('--epochs', type=int, default=5 if question == 1 else 50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--augment', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--run-name', default=f'q{question}')
    parser.add_argument('--data-root', type=Path, default=COURSE_DIR / 'data' / 'VOCdevkit' / 'VOC2007')
    parser.add_argument('--log-root', type=Path, default=COURSE_DIR / 'runs')
    parser.add_argument('--checkpoint-root', type=Path, default=COURSE_DIR / 'checkpoints')
    cli = parser.parse_args()
    return ARGS(epochs=cli.epochs, inp_size=64 if question == 1 else 224,
                use_cuda=True, lr=cli.lr, batch_size=cli.batch_size,
                step_size=3 if question == 1 else 15,
                gamma=0.5 if question == 1 else 0.3, val_every=70,
                log_every=10, augment=cli.augment, seed=cli.seed,
                run_name=cli.run_name, data_root=str(cli.data_root),
                log_root=str(cli.log_root), checkpoint_root=str(cli.checkpoint_root))
