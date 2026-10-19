"""Project fine-tuned Q2 features and blend colors of active GT classes."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.manifold import TSNE
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'q1_q2_classification'))
from train_q2 import ResNet
from voc_dataset import VOCDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default='q2_lr0001')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(8)
    course = ROOT.parent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(course/'checkpoints'/args.run_name/'best.pt', map_location='cpu')
    model = ResNet(20).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    dataset = VOCDataset('test',224,str(course/'data/VOCdevkit/VOC2007'),augment=False)
    indices = np.random.default_rng(args.seed).choice(len(dataset),1000,replace=False)
    loader = DataLoader(Subset(dataset,indices),batch_size=128,num_workers=4)
    features, labels = [], []
    captured = []
    hook = model.resnet.avgpool.register_forward_hook(lambda module, inputs, output: captured.append(output.flatten(1).cpu()))
    with torch.no_grad():
        for images, targets, weights in loader:
            model(images.to(device))
            features.append(captured.pop())
            labels.append(targets)
    hook.remove()
    features = torch.cat(features).numpy()
    labels = torch.cat(labels).numpy()
    projection = TSNE(n_components=2,perplexity=30,init='pca',learning_rate='auto',
                      max_iter=1500,random_state=args.seed).fit_transform(features)
    base_colors = np.array(plt.get_cmap('tab20').colors)
    colors = labels @ base_colors / labels.sum(axis=1,keepdims=True)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(projection[:,0],projection[:,1],c=colors,s=13,alpha=.8,linewidths=0)
    ax.set(xlabel='t-SNE 1',ylabel='t-SNE 2')
    ax.set_xticks([]); ax.set_yticks([])
    legend = [Line2D([],[],marker='o',linestyle='',color=color,label=name,markersize=5)
              for color,name in zip(base_colors,dataset.CLASS_NAMES)]
    ax.legend(handles=legend,bbox_to_anchor=(1.01,1),loc='upper left',frameon=False,fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT/'report/figures/q2_tsne.png',dpi=180)
    plt.close(fig)
    output = course/'runs'/args.run_name
    np.savez_compressed(output/'tsne.npz',indices=indices,features=features,labels=labels,
                        projection=projection,colors=colors)
    (output/'tsne_config.json').write_text(json.dumps(dict(seed=args.seed,n_images=1000,
        checkpoint_epoch=checkpoint['epoch'],feature_dim=512,perplexity=30,
        max_iter=1500,init='pca',learning_rate='auto'),indent=2))
    print('Saved t-SNE of',features.shape,'features from checkpoint epoch',checkpoint['epoch'])


if __name__ == '__main__':
    main()
