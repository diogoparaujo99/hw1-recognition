"""Create report plots from recorded TensorBoard events and evaluator output."""
import argparse
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT.parent / 'runs'
FIGURES = ROOT / 'report/figures'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.grid':True,'grid.alpha':.18,'figure.dpi':120})


def events(name):
    return EventAccumulator(str(RUNS/name),size_guidance={'scalars':0,'histograms':0}).Reload()


def scalar_plot(runs,tag,filename,ylabel):
    fig,ax=plt.subplots(figsize=(7,3.2))
    for run,label in runs:
        values=events(run).Scalars(tag)
        ax.plot([v.step for v in values],[v.value for v in values],label=label,lw=1.6)
    ax.set(xlabel='Training step',ylabel=ylabel)
    if len(runs)>1:
        ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(FIGURES/filename,dpi=170);plt.close(fig)


def q1():
    names=['q1_lr001_noaug','q1_lr001_aug','q1_lr0003_noaug','q1_lr0003_aug']
    histories={name:json.loads((RUNS/name/'history.json').read_text()) for name in names}
    best=max(names,key=lambda name:max(row['map'] for row in histories[name]))
    prefix='q1_lr001' if 'lr001_' in best else 'q1_lr0003'
    pairs=[(prefix+'_noaug','Without augmentation'),(prefix+'_aug','With augmentation')]
    scalar_plot(pairs,'Loss/Train','q1_compare_loss.png','Training loss')
    scalar_plot(pairs,'mAP','q1_compare_map.png','Test mAP')
    for tag,name,label in [('Loss/Train','loss','Training loss'),('mAP','map','Test mAP'),
                           ('learning_rate','lr','Learning rate')]:
        scalar_plot([(best,best)],tag,f'q1_{name}.png',label)
    (RUNS/'selected_q1.json').write_text(json.dumps(dict(run=best,history=histories),indent=2))
    print('Selected Q1:',best)


def histogram_plot(run,tag,filename):
    values=events(run).Histograms(tag)
    selected=[values[0],values[len(values)//2],values[-1]]
    fig,ax=plt.subplots(figsize=(7,3.2))
    # TensorBoard uses unequal source buckets. Rebin by interval overlap so
    # differing bucket widths do not create misleading peaks around zero.
    lower=min(item.histogram_value.min for item in selected)
    upper=max(item.histogram_value.max for item in selected)
    edges=np.linspace(lower,upper,81)
    for item in selected:
        h=item.histogram_value
        counts=np.array(h.bucket,dtype=float)
        limits=np.array(h.bucket_limit)
        lows=np.maximum(np.r_[-np.inf,limits[:-1]],h.min)
        highs=np.minimum(limits,h.max)
        rebinned=np.zeros(80)
        for low,high,count in zip(lows,highs,counts):
            if count==0:
                continue
            if high<=low:
                index=np.clip(np.searchsorted(edges,low,side='right')-1,0,79)
                rebinned[index]+=count
            else:
                overlaps=np.maximum(0,np.minimum(edges[1:],high)-np.maximum(edges[:-1],low))
                rebinned+=count*overlaps/(high-low)
        assert np.isclose(rebinned.sum(),h.num)
        ax.stairs(rebinned/h.num/np.diff(edges),edges,label=f'Step {item.step}',lw=1.4)
    ax.set(xlabel='Gradient',ylabel='Probability density')
    ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
    ax.legend(frameon=False);fig.tight_layout();fig.savefig(FIGURES/filename,dpi=170);plt.close(fig)


def q2():
    run='q2_lr0001'
    for tag,name,label in [('Loss/Train','loss','Training loss'),('mAP','map','Test mAP'),
                           ('learning_rate','lr','Learning rate')]:
        scalar_plot([(run,run)],tag,f'q2_{name}.png',label)
    histogram_plot(run,'resnet.layer1.1.conv1.weight/grad','q2_hist_conv.png')
    histogram_plot(run,'resnet.layer4.0.bn2.bias/grad','q2_hist_bn.png')


def q3():
    for run,name in [('fcos_overfit','q3_overfit.png'),('fcos_full','q3_full.png')]:
        h=json.loads((RUNS/run/'history.json').read_text())
        fig,ax=plt.subplots(figsize=(7,3.2))
        ax.plot([v['iteration'] for v in h],[v['total'] for v in h],lw=.65,alpha=.6,label='Total loss')
        ax.set(xlabel='Iteration',ylabel='Training loss')
        fig.tight_layout();fig.savefig(FIGURES/name,dpi=170);plt.close(fig)
    results=(ROOT.parent/'tools/mAP/output/output.txt').read_text()
    rows=re.findall(r'([0-9.]+)% = (\w+) AP',results)
    mean=float(re.search(r'mAP = ([0-9.]+)%',results)[1])
    fig,ax=plt.subplots(figsize=(8,4.5))
    names=[v[1] for v in rows]+['mAP']
    vals=[float(v[0]) for v in rows]+[mean]
    ax.bar(names,vals,color=['#527f9d']*len(rows)+['#bc603d'])
    ax.set(ylabel='AP at IoU 0.5 (%)',ylim=(0,100))
    ax.tick_params(axis='x',rotation=60,labelsize=10)
    fig.tight_layout();fig.savefig(FIGURES/'q3_map.png',dpi=180);plt.close(fig)
    (RUNS/'detection_metrics.json').write_text(json.dumps(dict(map=mean,ap=dict(zip(names,vals))),indent=2))
    print('Detection mAP:',mean)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--question',choices=['q1','q2','q3','all'],default='all')
    args=parser.parse_args();FIGURES.mkdir(exist_ok=True)
    for name,fn in [('q1',q1),('q2',q2),('q3',q3)]:
        if args.question in [name,'all']:
            fn()


if __name__=='__main__':
    main()
