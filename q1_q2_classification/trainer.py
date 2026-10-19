from __future__ import print_function

import torch
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import utils
from voc_dataset import VOCDataset
from simple_cnn import SimpleCNN


def save_this_epoch(args, epoch):
    if args.save_freq > 0 and (epoch+1) % args.save_freq == 0:
        return True
    if args.save_at_end and (epoch+1) == args.epochs:
        return True
    return False


def save_model(epoch, model_name, model):
    filename = 'checkpoint-{}-epoch{}.pth'.format(
        model_name, epoch+1)
    checkpoint_dir = os.environ.get('VLR_CHECKPOINT_ROOT',
        os.path.join(os.path.dirname(__file__), '../../checkpoints'))
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = os.path.join(checkpoint_dir, filename)
    print("saving model at ", filename)
    torch.save(model, filename)


def train(args, model, optimizer, scheduler=None, model_name='model'):
    log_root = os.environ.get('VLR_LOG_ROOT',
        os.path.join(os.path.dirname(__file__), '../../runs'))
    writer = SummaryWriter(os.path.join(log_root, model_name))
    train_loader = utils.get_data_loader(
        'voc', train=True, batch_size=args.batch_size, split='trainval', inp_size=args.inp_size)
    test_loader = utils.get_data_loader(
        'voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)

    # Ensure model is in correct mode and on right device
    model.train()
    model = model.to(args.device)

    cnt = 0

    for epoch in range(args.epochs):
        for batch_idx, (data, target, wgt) in enumerate(train_loader):
            data, target, wgt = data.to(args.device), target.to(args.device), wgt.to(args.device)

            optimizer.zero_grad()
            output = model(data)

            ##################################################################
            # TODO: Implement a suitable loss function for multi-label
            # classification. You are NOT allowed to use any pytorch built-in
            # functions. Remember to take care of underflows / overflows.
            # Function Inputs:
            #   - `output`: Outputs from the network
            #   - `target`: Ground truth labels, refer to voc_dataset.py
            #   - `wgt`: Weights (difficult or not), refer to voc_dataset.py
            # Function Outputs:
            #   - `output`: Computed loss, a single floating point number
            ##################################################################
            if isinstance(model, SimpleCNN):
                # Q1 uses the built-in loss, as permitted by the assignment.
                elementwise_loss = torch.nn.BCEWithLogitsLoss(reduction="none")(output, target)
            else:
                elementwise_loss = (torch.maximum(output, torch.zeros_like(output))
                                    - output * target
                                    + torch.log1p(torch.exp(-output.abs())))
            loss = (elementwise_loss * wgt).sum() / wgt.sum().clamp_min(1)
            ##################################################################
            #                          END OF YOUR CODE                      #
            ##################################################################
            
            loss.backward()
            
            if cnt % args.log_every == 0:
                writer.add_scalar("Loss/train", loss.item(), cnt)
                print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, cnt, 100. * batch_idx / len(train_loader), loss.item()))
                
                # Log gradients. For the Q2 handout's requested histograms:
                # "Conv1 Layer": layer1.1.conv1.weight
                # "BN4 layer": layer4.0.bn2.bias
                # With the ResNet wrapper, TensorBoard tags are
                # resnet.layer1.1.conv1.weight/grad and
                # resnet.layer4.0.bn2.bias/grad.
                for tag, value in model.named_parameters():
                    if value.grad is not None:
                        writer.add_histogram(tag + "/grad", value.grad.cpu().numpy(), cnt)

            optimizer.step()
            
            # Validation iteration
            if cnt % args.val_every == 0:
                model.eval()
                ap, map = utils.eval_dataset_map(model, args.device, test_loader)
                print("map: ", map)
                writer.add_scalar("map", map, cnt)
                model.train()
            
            cnt += 1

        if scheduler is not None:
            scheduler.step()
            writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], cnt)

        # save model
        if save_this_epoch(args, epoch):
            save_model(epoch, model_name, model)

    # Validation iteration
    model.eval()
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)
    ap, map = utils.eval_dataset_map(model, args.device, test_loader)
    writer.add_scalar("map", map, cnt)
    writer.close()
    return ap, map
