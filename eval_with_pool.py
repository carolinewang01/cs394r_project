import pprint
import time
import glob
import numpy as np
from datetime import timedelta
import os
import ray
import json

@ray.remote(num_gpus=1./20)
class Actor(object):
    def __init__(self,
                log_dir,
                agent_resume_path,
                opponents_pool_path):
        self.log_dir = log_dir
        self.agent_resume_path = agent_resume_path
        self.opponents_pool_path = opponents_pool_path
    
    def evaluate_with_pool(self):
            exp_tag = self.agent_resume_path.split('/')[-3].split('_')[-1]
            result_file = self.log_dir+'/'+self.agent_resume_path.split('/')[-3]+'.json'
            if exp_tag not in results.keys():
                results[exp_tag] = {}
                results[exp_tag]['rews'] = []
                results[exp_tag]['winrates'] = []
                results[exp_tag]['wintierates'] = []
                aggr_results[exp_tag] = {}
            rews=[]
            winrates=[]
            wintierates=[]
            for opponent in glob.glob(self.opponents_pool_path+'iqn-vs-random_trial=[0-3][0-9]*=pow'):
                opponent_resume_path = opponent+'/policy.pth'
                print('Main Agent:', self.agent_resume_path)
                print('Opponent:',opponent_resume_path)
                try:
                    rew, winrate,wintierate = test_pool(
                            agent_resume_path=self.agent_resume_path,
                            opponent_resume_path=opponent_resume_path)
                except:
                    end = time.time()
                    elapsed = str(timedelta(seconds=end - start))
                    print("SCRIPT RUN TIME: ", elapsed)
                    print('mean reward:',np.mean(rews), ' std:', np.std(rews), ' winrate:', np.mean(winrates)*100, ' wintie rate:', np.mean(wintierates)*100, ' num of seeds:', len(rews))
                    break
                if rew is not None:
                    rews.append(rew)
                    winrates.append(winrate)
                    wintierates.append(wintierate)
                    end = time.time()
                    elapsed = str(timedelta(seconds=end - start))
                    print("SCRIPT RUN TIME: ", elapsed)
                    print('mean reward:',np.mean(rews), ' std:', np.std(rews), ' winrate:', np.mean(winrates)*100, ' wintie rate:', np.mean(wintierates)*100, ' num of seeds:', len(rews))
            results[exp_tag]['rews'].append(np.mean(rews))
            results[exp_tag]['winrates'].append(np.mean(winrates)*100)
            results[exp_tag]['wintierates'].append(np.mean(wintierates)*100)
            aggr_results[exp_tag]['mean_rews']=np.mean(results[exp_tag]['rews'])
            aggr_results[exp_tag]['mean_winrates']=np.mean(results[exp_tag]['winrates'])
            aggr_results[exp_tag]['mean_wintierates']=np.mean(results[exp_tag]['wintierates'])
            aggr_results[exp_tag]['std_rews']=np.std(results[exp_tag]['rews'])
            aggr_results[exp_tag]['std_winrates']=np.std(results[exp_tag]['winrates'])
            aggr_results[exp_tag]['std_wintierates']=np.std(results[exp_tag]['wintierates'])

            print(aggr_results)
            json.dump(aggr_results, open(result_file,'w'))



def test_pool(
            env_id='leduc',
            agent_learn_algo='iqn',
            agent_resume_path=None,
            opponent_algo='iqn',
            opponent_resume_path=None,
            num_eval_episodes=100
            ):
    
    # from train_vs_random import get_args, watch
    from train_risk_aware import get_args, watch

    args = get_args()
    
    args.env_id = env_id
    args.agent_learn_algo = agent_learn_algo
    args.agent_resume_path = agent_resume_path
    args.opponent_learn_algo = opponent_algo
    args.opponent_resume_path = opponent_resume_path
    args.num_eval_episodes = num_eval_episodes 
    return watch(args)

if __name__ == '__main__':
    SEEDS = np.load('seeds.npy')
    ENV_IDS = ["leduc", 
               "texas",
               ]
    start = time.time()
    
    rews=[]
    winrates=[]
    wintierates=[]

    env_id = "leduc"
    agent_learn_algo = "dqn" #iqn
    # trial_idx = 0
    eta = 1.0 # 1.5
    risk_distortion = "cvar" #  None # "pow"

    agents_pool_path = "log/leduc/"
    opponents_pool_path = 'log/leduc/'
    np.random.seed(112358)
    n_opponents = 32

    # agent_pool_list = glob.glob(agents_pool_path+f'{agent_learn_algo}-vs-random_trial=*_eta={eta}_risk-distort={risk_distortion}')
    agent_pool_list = glob.glob(agents_pool_path+f'{agent_learn_algo}-vs-iqn_trial=*_opponent-eta={eta}_opponent-risk-distort={risk_distortion}')

    opponents_pool_list = np.random.choice(glob.glob(opponents_pool_path+'iqn-vs-random_trial=*=pow'), n_opponents)

    rew, winrate, wintierate = None, None, None
    for agent in agent_pool_list:
        agent_resume_path = agent+'/policy.pth'
        for opponent in opponents_pool_list:
            opponent_resume_path = opponent+'/policy.pth'
            print('Main Agent:', agent_resume_path)
            print('Opponent:',opponent_resume_path)
            try:
                rew, winrate,wintierate = test_pool(
                                            env_id=env_id,
                                            agent_learn_algo=agent_learn_algo,
                                            agent_resume_path=agent_resume_path,
                                            opponent_algo='iqn',
                                            opponent_resume_path=opponent_resume_path
                                            )
            except Exception as e:
                print(e)
                end = time.time()
                elapsed = str(timedelta(seconds=end - start))
                print("EVAL RUN TIME: ", elapsed)
                print('mean reward:',np.mean(rews), ' std:', np.std(rews), ' winrate:', np.mean(winrates)*100, ' wintie rate:', np.mean(wintierates)*100, ' num of evals:', len(rews))
                break

            if rew is not None:
                rews.append(rew)
                winrates.append(winrate)
                wintierates.append(wintierate)
                end = time.time()
                elapsed = str(timedelta(seconds=end - start))
                print("EVAL RUN TIME: ", elapsed)
                print('mean reward:',np.mean(rews), ' std:', np.std(rews), ' winrate:', np.mean(winrates)*100, ' wintie rate:', np.mean(wintierates)*100, ' num of evals:', len(rews))
    
    '''
    # EVAL SELF PLAY AGAINST POOL
    opponents_pool_path = 'log/leduc-pool/'
    rew, winrate, wintierate = None, None, None
    results = {} 
    aggr_results = {}
    log_dir = 'evaluate_results'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    RISK_AWARE = [True, False]
    ray.init(num_gpus=1,local_mode=False)
    MAX_NUM_JOB=6
    srs=[]
    for trial_idx, seed  in enumerate(SEEDS):
        if trial_idx < 80:continue
        for risk_aware in RISK_AWARE:
            agent_resume_path = 'log/selfplay/leduc/iqn-selfplay_trial={}_riskaware={}/9/policy.pth'.format(trial_idx,risk_aware)
            srs.append(Actor.remote(log_dir,
                                    agent_resume_path,
                                    opponents_pool_path))
            if len(srs)>=MAX_NUM_JOB:
                jobs=[]
                for sr in srs:
                    jobs.append(sr.evaluate_with_pool.remote())
                ray.wait(jobs, num_returns=len(jobs))
                for sr in srs:
                    ray.kill(sr)
                del srs
                srs=[]

    '''
    end = time.time()
    elapsed = str(timedelta(seconds=end - start))
    print("SCRIPT RUN TIME: ", elapsed)
