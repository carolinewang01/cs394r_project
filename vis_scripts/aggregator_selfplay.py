import os
from collections import defaultdict
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from matplotlib import pyplot as plt
import glob
import seaborn
seaborn.set_theme(font_scale=2.0)

def plot_single_exp(filelists,
                    legend=None):
    summary_iterators = [EventAccumulator(f).Reload() for f in filelists]
    valid_summary_iterators = []
    tags = summary_iterators[0].Tags()['scalars']
    for it in summary_iterators:
        if it.Tags()['scalars'] == tags:
            valid_summary_iterators.append(it)
    legend = legend.split('_')
    '''
    eta = legend[1].split('+')
    eta_self = eta[0]
    eta_oppo = eta[1]
    '''
    try:
        eta = float(legend[1])
        if legend[-1]=='None' or eta==0:
            legend = 'Risk-Neutral IQN vs DQN'
        elif eta>0:
            legend = 'Risk-Seeking IQN vs DQN (eta={})'.format(eta)
        elif eta<0:
            legend = 'Risk-Averse IQN vs DQN (eta={})'.format(eta)
    except:
        eta = legend[1].split('+')
        eta_self = float(eta[0])
        eta_oppo = float(eta[1])
        legend='Self:{}, Opponent:{}'.format(eta_self, eta_oppo)

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
    plt.figure(0, figsize=(10,7.5))
    plt.rc('legend',fontsize='medium')
    # Sort the order of the experiments
    keys = sorted(filelists.keys(), key=lambda x: abs(float(x.split('_')[1].split('+')[-1])))
    for key in keys:
        attr = key
        files = filelists[key]
        plot_single_exp(filelists=files, legend=attr)
    plt.legend()
    #plt.title('Risk-Seeking v.s. Risk-Averse')
    plt.ylabel('Return')
    plt.xlabel('steps')
    plt.ylim(-1.5,2.0)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.show()

def aggregate(
        exp_path='./log/leduc/',
        exp_tags=['iqn-vs-random','dqn-vs-random'],
        exp_risk_distortions=['cvar', 'None'],
        etas=[]
        ):
    print(exp_path)
    expdirs=glob.glob(exp_path+'/*')
    filelists={}
    print(expdirs)
    for ed in expdirs:
        expconfig=ed.split('/')[-1].split('_')
        exp_tag=expconfig[0]
        exp_trial=expconfig[1].split("=")[-1]
        exp_eta=expconfig[2].split("=")[-1]
        exp_risk_distortion=expconfig[3].split("=")[-1]
        attr='_'.join([exp_tag, exp_eta, exp_risk_distortion])
        print(attr) 
        if exp_tag in exp_tags \
            and exp_risk_distortion in exp_risk_distortions \
            and ((len(etas)>0 and exp_eta in etas) or exp_risk_distortion=='None'):
            if attr not in filelists.keys():
                filelists[attr]=[]
            filelists[attr].append(glob.glob(ed+'/events*')[0])
    print(filelists.keys())
    plot_exps(filelists)

if __name__=='__main__':
    aggregate(
            exp_path=os.path.expanduser('~')+'/Documents/cs394r_project/log/leduc/',
            exp_tags=['iqn-vs-iqn'],
            exp_risk_distortions=['cvar','None','pow'], 
            etas=['0.2+0.5','0.2+1.0','0.2+1.5'])
