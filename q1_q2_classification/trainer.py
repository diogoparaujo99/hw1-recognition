"""Training and logging shared by the two multi-label classifiers."""
import json
from pathlib import Path
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import utils


def masked_binary_cross_entropy(output, target, wgt):
    """Stable BCE from logits; difficult image/class labels do not contribute."""
    positive = torch.maximum(output, torch.zeros_like(output))
    elementwise = positive - output * target + torch.log1p(torch.exp(-output.abs()))
    return (elementwise * wgt).sum() / wgt.sum().clamp_min(1)


def save_this_epoch(args, epoch):
    return ((args.save_freq > 0 and (epoch + 1) % args.save_freq == 0)
            or (args.save_at_end and epoch + 1 == args.epochs))


def save_model(epoch, model_name, model, directory='../checkpoints'):
    path = Path(directory) / f'{model_name}-epoch{epoch + 1}.pth'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def train(args, model, optimizer, scheduler=None, model_name='model'):
    run_dir = Path(args.log_root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_root) / args.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    settings = {k: getattr(args, k) for k in dir(args)
                if not k.startswith('_') and k != 'device' and not callable(getattr(args, k))}
    (run_dir / 'config.json').write_text(json.dumps(settings, indent=2))
    writer = SummaryWriter(str(run_dir))
    train_loader = utils.get_data_loader(batch_size=args.batch_size, split='trainval',
        inp_size=args.inp_size, data_root=args.data_root, augment=args.augment)
    test_loader = utils.get_data_loader(train=False, batch_size=args.test_batch_size,
        split='test', inp_size=args.inp_size, data_root=args.data_root, augment=False)
    model = model.to(args.device)
    history, best_map, step = [], -1.0, 0
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        total_loss, count = 0.0, 0
        epoch_lr = optimizer.param_groups[0]['lr']
        for data, target, wgt in train_loader:
            data, target, wgt = [v.to(args.device, non_blocking=True) for v in (data, target, wgt)]
            optimizer.zero_grad(set_to_none=True)
            loss = masked_binary_cross_entropy(model(data), target, wgt)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at step {step}')
            loss.backward()
            if step % args.log_every == 0:
                writer.add_scalar('Loss/Train', loss.item(), step)
                for name, parameter in model.named_parameters():
                    if name in ('resnet.layer1.1.conv1.weight', 'resnet.layer4.0.bn2.bias'):
                        writer.add_histogram(name + '/grad', parameter.grad.detach(), step)
            optimizer.step()
            total_loss += loss.item() * data.shape[0]
            count += data.shape[0]
            step += 1
        model.eval()
        ap, map_value = utils.eval_dataset_map(model, args.device, test_loader)
        writer.add_scalar('mAP', map_value, step)
        writer.add_scalar('learning_rate', epoch_lr, step)
        row = dict(epoch=epoch + 1, step=step, loss=total_loss/count,
                   map=float(map_value), lr=epoch_lr, elapsed=time.time()-start)
        history.append(row)
        (run_dir / 'history.json').write_text(json.dumps(history, indent=2))
        if map_value > best_map:
            best_map = float(map_value)
            torch.save({'model': model.state_dict(), 'epoch': epoch + 1,
                        'map': best_map, 'config': settings}, checkpoint_dir / 'best.pt')
        print(json.dumps(row), flush=True)
        if scheduler is not None:
            scheduler.step()
    torch.save({'model': model.state_dict(), 'epoch': args.epochs,
                'map': float(map_value), 'config': settings}, checkpoint_dir / 'last.pt')
    writer.close()
    return ap, map_value
