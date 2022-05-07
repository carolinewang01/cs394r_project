import os
from collections import defaultdict
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from matplotlib import pyplot as plt
import glob
import seaborn
seaborn.set_theme()

def plot_single_exp(filelists,
                    legend=None):
    summary_iterators = [EventAccumulator(f).Reload() for f in filelists]
    valid_summary_iterators = []
    tags = summary_iterators[0].Tags()['scalars']
    for it in summary_iterators:
        if it.Tags()['scalars'] == tags:
            valid_summary_iterators.append(it)
  
    step = None
    out = defaultdict(list)
    std = defaultdict(list)
    for tag in ['test/reward']:#tags:
        data = []
        step = None
        for it in valid_summary_iterators:
            events = it.Scalars(tag)
            data.append([e.value for e in events])
            step=[e.step for e in events]
        data=np.array(data)
        out[tag]=np.mean(data,axis=0)
        std[tag]=np.std(data,axis=0)
        plt.plot(step,out[tag], label=legend)
        plt.fill_between(step, out[tag]-std[tag], out[tag]+std[tag], alpha=0.3)
    return out, std

def plot_exps(filelists):
    plt.figure(0)
    for attr, files in filelists.items():
        plot_single_exp(filelists=files, legend=attr)
    plt.legend()
    plt.title('IQN/DQN vs Random Agents')
    plt.ylabel('Reward')
    plt.xlabel('steps')
    plt.show()

def aggregate(
        exp_path='./log/leduc/',
        exp_tags=['iqn-vs-random','dqn-vs-random'],
        exp_risk_distortions=['cvar', 'None'],
        ):
    print(exp_path)
    expdirs=glob.glob(exp_path+'*')
    filelists={}
    for ed in expdirs:
        expconfig=ed.split('/')[-1].split('_')
        exp_tag=expconfig[0]
        exp_trial=expconfig[1].split("=")[-1]
        exp_eta=expconfig[2].split("=")[-1]
        exp_risk_distortion=expconfig[3].split("=")[-1]
        attr='_'.join([exp_tag, exp_eta, exp_risk_distortion])
        print(attr) 
        if exp_tag in exp_tags \
            and exp_risk_distortion in exp_risk_distortions:
            if attr not in filelists.keys():
                filelists[attr]=[]
            filelists[attr].append(glob.glob(ed+'/events*')[0])
    print(filelists.keys())
    plot_exps(filelists)

if __name__=='__main__':
    aggregate()
