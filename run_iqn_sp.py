import ray 
import pprint
import time
import numpy as np
from datetime import timedelta

@ray.remote(num_gpus=1./20)
class Actor(object):
    def __init__(self,
                env_id, 
                agent_learn_algo, 
                eta=1.0, risk_distortion=None,
                seed=1626, trial_idx=0,
                risk_aware=False
    ):
        self.env_id = env_id
        self.agent_learn_algo = agent_learn_algo
        self.eta = eta
        self.risk_distortion=risk_distortion
        self.seed = seed
        self.trial_idx = trial_idx
        self.risk_aware = risk_aware

    def train_sp(self):
        '''train args.algo agent vs pre-trained iqn agent

        Make sure to specify --agent-learn-algo, --eta, --risk-distortion, --opponent-resume-path
        '''
        from train_iqn_sp import get_args, train_agent, watch, train_selfplay

        args = get_args()
        args.env_id = self.env_id
        args.agent_learn_algo = self.agent_learn_algo

        args.eta = self.eta
        args.risk_distortion = self.risk_distortion
        args.seed = self.seed
        args.trial_idx = self.trial_idx
        args.risk_aware = self.risk_aware
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

def main():
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
    MAX_NUM_JOB=2
    if EXPT_NAME == "train_sp":
        ray.init(logging_level=0, num_gpus=1, local_mode=False)
        srs=[]
        for env_id in ENV_IDS:
            for trial_idx, seed in enumerate(SEEDS):
                if trial_idx<99:continue
                for risk_aware in RISK_AWARE:
                    srs.append(Actor.remote(env_id=env_id, 
                                        agent_learn_algo="iqn",
                                        eta=1.0, risk_distortion=None,
                                        seed=int(seed), trial_idx=trial_idx,
                                        risk_aware=risk_aware))

                    if len(srs)>=MAX_NUM_JOB:
                        jobs=[]
                        for sr in srs:
                            jobs.append(sr.train_sp.remote())
                        ray.wait(jobs, num_returns=len(jobs))
                        for sr in srs:
                            ray.kill(sr)
                        del srs
                        srs=[]
    elif EXPT_NAME == "test_sp":
            rews=[]
            winrates=[]
            wintierates=[]
            for trial_idx, seed in enumerate(SEEDS):
                rew, winrate, wintierate = None, None, None
                if trial_idx<80:continue
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



if __name__ == '__main__':
    import sys
    sys.exit(main())
