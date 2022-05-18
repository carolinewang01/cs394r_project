import ray 
import pprint
import time
import numpy as np
from datetime import timedelta

@ray.remote(num_gpus=1./20)
def train_sp(env_id, 
                     agent_learn_algo, 
                     eta=1.0, risk_distortion=None,
                     seed=1626, trial_idx=0,
                     risk_aware=False
    ):
    '''train args.algo agent vs pre-trained iqn agent

    Make sure to specify --agent-learn-algo, --eta, --risk-distortion, --opponent-resume-path
    '''
    from train_iqn_sp import get_args, train_agent, watch, train_selfplay

    args = get_args()
    args.env_id = env_id
    args.agent_learn_algo = agent_learn_algo

    args.eta = eta
    args.risk_distortion = risk_distortion
    args.seed = seed
    args.trial_idx = trial_idx
    args.risk_aware=risk_aware
    result, agent = train_selfplay(args)
    pprint.pprint(result)

def test_sp(
            env_id='leduc',
            agent_learn_algo='iqn',
            agent_resume_path=None,
            opponent_algo='iqn',
            opponent_resume_path=None,
            ):
    
    from train_iqn_sp import get_args, watch
    args = get_args()
    
    args.env_id = env_id
    args.agent_learn_algo = agent_learn_algo
    args.agent_resume_path = agent_resume_path
    args.opponent_learn_algo = opponent_algo
    args.opponent_resume_path = opponent_resume_path
    
    return watch(args)


if __name__ == '__main__':
    SEEDS = [1626, 174, 571, 2948, 109284]
    SEEDS = np.load('seeds.npy')
    ENV_IDS = [ "leduc", 
                #"texas",
               # "texas-no-limit" # order of agents fixed, need to fix this
               ]
    
    EXPT_NAME = "train_sp" #"train_sp_risk_aware" # "train_sp"
    RISK_AWARE = [True, False]
    ##################################################
    start = time.time()
    MAX_NUM_JOB=20
    if EXPT_NAME == "train_sp":
        import logging
        ray.init(logging_level=logging.INFO, num_gpus=1)
        jobs=[]
        for env_id in ENV_IDS:
            for trial_idx, seed in enumerate(SEEDS):
                if trial_idx<30:continue
                for risk_aware in RISK_AWARE:
                    '''
                    jobs.append(train_sp(env_id=env_id, 
                                        agent_learn_algo="iqn",
                                        eta=1.0, risk_distortion=None,
                                        seed=int(seed), trial_idx=trial_idx,
                                        risk_aware=risk_aware).remote())
                    '''
                    jobs.append(train_sp.remote(env_id=env_id, 
                                        agent_learn_algo="iqn",
                                        eta=1.0, risk_distortion=None,
                                        seed=int(seed), trial_idx=trial_idx,
                                        risk_aware=risk_aware))
                    if len(jobs)>=MAX_NUM_JOB:
                        ray.wait(jobs, num_returns=len(jobs))
                        for job in jobs:
                            ray.kill(job)
                        del jobs
                        jobs=[]
                        time.sleep(10)
    elif EXPT_NAME == "test_sp":
            rews=[]
            winrates=[]
            wintierates=[]
            for trial_idx, seed in enumerate(SEEDS):
                rew, winrate, wintierate = None, None, None
                try:
                    agent_resume_path = 'log/selfplay/leduc/iqn-selfplay_trial={}_riskaware=True/9/policy.pth'.format(trial_idx)
                    opponent_resume_path = 'log/selfplay/leduc/iqn-selfplay_trial={}_riskaware=False/9/policy.pth'.format(trial_idx)

                    rew, winrate,wintierate = test_sp(
                            agent_resume_path=agent_resume_path,
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
                print('mean reward:',np.mean(rews), ' std:', np.std(rews), ' winrate:', np.mean(winrates)*100, ' wintie rate:', np.mean(wintierates)*100, ' num of seeds:', len(rews))
    end = time.time()
    elapsed = str(timedelta(seconds=end - start))
    print("SCRIPT RUN TIME: ", elapsed)
