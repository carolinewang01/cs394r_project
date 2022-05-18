import os
from collections import defaultdict, OrderedDict
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from matplotlib import pyplot as plt
import glob
import seaborn
seaborn.set_theme()
seaborn.set(font_scale=1.2)

def plot_single_exp(filelists,
                    legend=None):
    summary_iterators = [EventAccumulator(f).Reload() for f in filelists]
    valid_summary_iterators = []
    tags = summary_iterators[0].Tags()['scalars']
    for it in summary_iterators:
        if it.Tags()['scalars'] == tags:
            valid_summary_iterators.append(it)
    assert len(summary_iterators) == len(valid_summary_iterators)
  
    step = None
    out = defaultdict(list)
    std = defaultdict(list)
    for tag in ['test/reward']:#tags:
        data = []
        step = None
        for i, it in enumerate(valid_summary_iterators):
            events = it.Scalars(tag)
            rewards = [e.value for e in events]
            if len(rewards)< 100:
                print("REWARDS SHAPE IS ", len(rewards), "FILENAME IS ", filelists[i])
            data.append(rewards)
            step=[e.step for e in events]

        data=np.array(data)
        out[tag]=np.mean(data,axis=0)
        print("DATA SHAPE IS ", data.shape)
        std[tag]=np.std(data,axis=0)
        plt.plot(step,out[tag], label=legend)
        plt.fill_between(step, out[tag]-std[tag], out[tag]+std[tag], alpha=0.3)
    return out, std

def plot_train_vs_random_exps(filelists, env_id, risk_type, save=False, savepath=None):
    plt.figure(0)
    for attr in sorted(filelists):
        files = filelists[attr]
        algos_names, eta, risk_distortion = attr.split("_")
        agent_algo = algos_names.split("-")[0]
        plot_single_exp(filelists=files, legend=f"{agent_algo}_eta={eta}")

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
    plt.title(f'IQN/DQN vs Random Agents on {env_id.capitalize()}, Risk Type={risk_type}')
    plt.ylabel('return')
    plt.xlabel('steps')
    if save:
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()

def plot_train_vs_iqn_pool_exps(filelists, env_id, risk_type, save=False, savepath=None):
    plt.figure(0)

    accum_filelists = {} # accumulate all files corresponding to same risk type
    for attr, filelist in filelists.items():        
        algos_names, eta, risk_distortion = attr.split("_")
        key = f"{algos_names}_{risk_distortion}"
        # if eta in ["0.4", "0.6", "0.8", "1.0"]:
        #     print("eta is ", eta)
        #     continue
        if key not in accum_filelists:
            accum_filelists[key] = []
        accum_filelists[key] += filelist

    for key in sorted(accum_filelists):
        print("PLOTTING KEY ", key)
        filelist = accum_filelists[key]
        algos_names = key.split("_")[0]
        agent_algo, _, opponent_algo = algos_names.split("-")
        plot_single_exp(filelists=filelist, legend=agent_algo)
    plt.legend()
    plt.title(f'IQN/DQN vs IQN Risk-Varying Agents on {env_id.capitalize()}, \nRisk Type={risk_type}')
    plt.ylabel('return')
    plt.xlabel('steps')
    if save:
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()

def aggregate(
        env_id='leduc',
        exp_tags=['iqn-vs-random','dqn-vs-random'],
        exp_risk_distortions=['cvar', 'None'],
        save=False
        ):
    exp_path = f"./log/{env_id}/"
    expdirs=glob.glob(exp_path+'*')
    filelists = {}
    for ed in expdirs:
        expconfig=ed.split('/')[-1].split('_')
        exp_tag=expconfig[0]
        exp_trial=expconfig[1].split("=")[-1]
        exp_eta=expconfig[2].split("=")[-1]
        exp_risk_distortion=expconfig[3].split("=")[-1]
        attr='_'.join([exp_tag, exp_eta, exp_risk_distortion])
        # print(attr) 
        if exp_tag in exp_tags \
            and exp_risk_distortion in exp_risk_distortions:
            if attr not in filelists.keys():
                filelists[attr]=[]
            filelists[attr].append(glob.glob(ed+'/events*')[0])
    print(filelists.keys())

    if exp_tags[0].split("-")[-1] == "random":
        savepath = f"./figures/iqn-dqn-vs-random_env-id={env_id}_risk-type={exp_risk_distortions[0]}.pdf"
        plot_train_vs_random_exps(filelists, env_id=env_id, risk_type=exp_risk_distortions[0], 
                  save=save, savepath=savepath)
    else:
        savepath = f"./figures/iqn-dqn-vs-iqn-pool_env-id={env_id}_risk-type={exp_risk_distortions[0]}.pdf"
        plot_train_vs_iqn_pool_exps(filelists, env_id=env_id, risk_type=exp_risk_distortions[0], 
                  save=save, savepath=savepath)

if __name__=='__main__':
    aggregate()
