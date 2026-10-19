import torch
import trainer
from utils import ARGS
from simple_cnn import SimpleCNN
from voc_dataset import VOCDataset
import numpy as np
import torchvision
import torch.nn as nn
import random
from config import parse_args


class ResNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()

        self.resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        ##################################################################
        # TODO: Define a FC layer here to process the features
        ##################################################################
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################
        

    def forward(self, x):
        ##################################################################
        # TODO: Return raw outputs here
        ##################################################################
        return self.resnet(x)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


if __name__ == "__main__":
    args = parse_args(2)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False

    print(args)

    ##################################################################
    # TODO: Define a ResNet-18 model (https://arxiv.org/pdf/1512.03385.pdf) 
    # Initialize this model with ImageNet pre-trained weights
    # (except the last layer). You are free to use torchvision.models 
    ##################################################################

    model = ResNet(len(VOCDataset.CLASS_NAMES)).to(args.device)

    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################

    # initializes Adam optimizer and simple StepLR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # trains model using your training code and reports test map
    test_ap, test_map = trainer.train(args, model, optimizer, scheduler, model_name=args.run_name)
    print('test map:', test_map)

    # Q2 t-SNE plot guidance:
    # Use this fine-tuned Q2 classifier in evaluation mode, not a fresh
    # ImageNet model. Extract the average pooling output immediately before
    # the final fully connected layer (resnet.avgpool), and flatten it to
    # one 512-dimensional feature vector per image for ResNet-18.
    # Use the 20 base PASCAL VOC classes (VOCDataset.CLASS_NAMES) in the
    # legend, rather than creating entries for multi-label combinations.
