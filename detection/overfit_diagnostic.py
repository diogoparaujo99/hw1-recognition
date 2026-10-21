"""Measure training and validation loss on the starter's small FCOS subset."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from detection_helper import inference_with_detector
from one_stage_detector import FCOS
from train import create_dataset_and_dataloaders

COURSE = Path(__file__).resolve().parents[2]


@torch.no_grad()
def measure_loss(model, loader, device, normalizer):
    """Use frozen batch statistics and one fixed loss scale for both splits."""
    previous_mode = model.training
    previous_normalizer = model._normalizer
    model.eval()
    model._normalizer = normalizer
    totals = {}
    count = 0
    try:
        for _, images, boxes in loader:
            losses = model(images.to(device), boxes.to(device))
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + value.item() * len(images)
            count += len(images)
    finally:
        model._normalizer = previous_normalizer
        model.train(previous_mode)
    result = {name: value / count for name, value in totals.items()}
    result['total'] = sum(result.values())
    return result


def measure_map(model, dataset, directory, device):
    """Run the supplied inference writer and an unchanged Cartucho evaluator."""
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(COURSE / 'tools/mAP/main.py', directory / 'main.py')
    loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True)
    inference_with_detector(model, loader, dataset.dataset.idx_to_class,
                            score_thresh=0.4, nms_thresh=0.6,
                            output_dir=str(directory / 'input'), device=device)
    with (directory / 'evaluator.log').open('w') as log:
        subprocess.run([sys.executable, str(directory / 'main.py'),
                        '--no-animation', '--no-plot', '--quiet'],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    text = (directory / 'output/output.txt').read_text()
    value = re.search(r'mAP = ([\d.]+)%', text)
    if value is None:
        raise RuntimeError('Cartucho did not produce an mAP result.')
    classes = re.findall(r'([\d.]+)% = (\S+) AP', text)
    return {'map_percent': float(value.group(1)),
            'class_ap_percent': {name: float(ap) for ap, name in classes}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--val-images', type=int, default=64)
    parser.add_argument('--run-name', default='fcos_160_val64_epoch400_seed0')
    args = parser.parse_args()
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run = COURSE / 'runs/q3_overfit_revision' / args.run_name
    checkpoints = COURSE / 'checkpoints/q3_overfit_revision' / args.run_name
    run.mkdir(parents=True, exist_ok=False)
    checkpoints.mkdir(parents=True, exist_ok=False)

    _, _, training, validation = create_dataset_and_dataloaders(subset=True)
    if not 0 < args.val_images <= len(validation):
        raise ValueError('--val-images must fit within the validation dataset.')
    val_indices = torch.linspace(0, len(validation) - 1, steps=args.val_images).long().tolist()
    held_out = Subset(validation, val_indices)
    train_paths = [training.dataset.instances[i][0] for i in training.indices]
    val_paths = [validation.instances[i][0] for i in val_indices]
    assert set(train_paths).isdisjoint(val_paths)
    loaders = [DataLoader(dataset, batch_size=16, num_workers=4,
                          pin_memory=True, persistent_workers=True)
               for dataset in (training, held_out)]
    model = FCOS(num_classes=20, fpn_channels=64, stem_channels=[64, 64]).to(device)
    normalizer = float(model._normalizer)
    iterations = args.epochs * len(loaders[0])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005,
                                momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(0.6 * iterations), int(0.9 * iterations)])
    sources = ['overfit_diagnostic.py', 'one_stage_detector.py',
               'detection_helper.py', 'detection_utils.py', 'train.py']
    settings = dict(command=[sys.executable] + sys.argv, cwd=os.getcwd(),
                    seed=0, epochs=args.epochs, iterations=iterations,
                    train_images=len(training), validation_images=len(held_out),
                    validation_to_training_ratio=len(held_out) / len(training),
                    validation_sampling='evenly_spaced',
                    train_indices=[int(i) for i in training.indices],
                    validation_indices=val_indices, train_paths=train_paths,
                    validation_paths=val_paths, batch_size=16, input_size=224,
                    optimizer='SGD', lr=0.005, momentum=0.9, weight_decay=1e-4,
                    scheduler_milestones=scheduler.milestones,
                    scheduler_gamma=0.1, evaluation_normalizer=normalizer,
                    score_threshold=0.4, nms_threshold=0.6, ap_iou=0.5,
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    torch_version=torch.__version__,
                    source_sha256={name: hashlib.sha256(
                        (Path(__file__).parent / name).read_bytes()).hexdigest()
                        for name in sources})
    (run / 'settings.json').write_text(json.dumps(settings, indent=2))
    writer = SummaryWriter(str(run))
    history = []
    maps = []
    updates = 0
    for epoch in range(args.epochs + 1):
        if epoch:
            model.train()
            update_losses = []
            for _, images, boxes in loaders[0]:
                losses = model(images.to(device), boxes.to(device))
                loss = sum(losses.values())
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Non-finite loss at update {updates}.')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                writer.add_scalar('train/update_total', loss.item(), updates)
                update_losses.append(loss.item())
                updates += 1
            writer.add_scalar('train/epoch_update_mean',
                              sum(update_losses) / len(update_losses), epoch)
        if epoch in (0, 1, 2) or epoch % 5 == 0 or epoch == args.epochs:
            before = {name: value.clone() for name, value in model.state_dict().items()}
            old_normalizer = model._normalizer
            train_loss = measure_loss(model, loaders[0], device, normalizer)
            val_loss = measure_loss(model, loaders[1], device, normalizer)
            assert model._normalizer == old_normalizer
            assert all(torch.equal(before[name], value)
                       for name, value in model.state_dict().items())
            record = dict(epoch=epoch, updates=updates,
                          train=train_loss, validation=val_loss,
                          lr=optimizer.param_groups[0]['lr'])
            history.append(record)
            (run / 'history.json').write_text(json.dumps(history, indent=2))
            for split, losses in [('train', train_loss), ('validation', val_loss)]:
                for name, value in losses.items():
                    writer.add_scalar(f'eval/{split}/{name}', value, epoch)
            writer.add_scalar('learning_rate', record['lr'], epoch)
            writer.flush()
            print(f"Epoch {epoch}: train={train_loss['total']:.6f}, "
                  f"validation={val_loss['total']:.6f}", flush=True)
        if epoch in (25, 100, args.epochs):
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'normalizer': model._normalizer},
                       checkpoints / f'epoch_{epoch}.pt')
            result = dict(epoch=epoch,
                          train=measure_map(model, training, run / f'ap_epoch{epoch}/train', device),
                          validation=measure_map(model, held_out, run / f'ap_epoch{epoch}/validation', device))
            maps.append(result)
            (run / 'map_results.json').write_text(json.dumps(maps, indent=2))
            for split in ('train', 'validation'):
                writer.add_scalar(f'mAP/{split}', result[split]['map_percent'], epoch)
            print('mAP:', json.dumps(result), flush=True)
    writer.close()
    print('Completed:', run, flush=True)


if __name__ == '__main__':
    main()
