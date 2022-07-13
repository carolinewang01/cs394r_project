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
                opponent_learn_algo,
                eta=1.0, opponent_eta=1.0, risk_distortion=None,
                seed=1626, trial_idx=0,
    ):
        self.env_id = env_id
        self.agent_learn_algo = agent_learn_algo
        self.opponent_learn_algo = opponent_learn_algo
        self.eta = eta
        self.opponent_eta = opponent_eta
        self.risk_distortion=risk_distortion
        self.seed = int(seed)
        self.trial_idx = trial_idx

    def train_iqn_iqn_indep(self): 
        from train_iqn_vs_iqn import get_args, train_agent, watch
        args = get_args()
        args.env_id = self.env_id
        args.agent_learn_algo = self.agent_learn_algo
        args.opponent_learn_algo = self.opponent_learn_algo
        args.eta = self.eta
        args.opponent_eta = self.opponent_eta
        args.risk_distortion = self.risk_distortion
        args.seed = self.seed
        args.trial_idx = self.trial_idx

        result, agent = train_agent(args)
        pprint.pprint(result)

def main():
    SEEDS = [1626, 174, 571, 2948, 109284]
    SEEDS = np.load('seeds.npy')
    ENV_IDS = [ #"leduc", 
                "texas",
               # "texas-no-limit" # order of agents fixed, need to fix this
               ]
 
    EXPT_NAME = "train_iqn_vs_iqn" #"train_sp_risk_aware" # "train_sp"
    SEEDS=np.load('seeds.npy')
    RISK_DISTORTION_DICT = {
        "pow":[-0.2,-0.2,-0.2, 0.2, 0.2, 0.2,0.2,0.2,0.2,-0.2,-0.2,-0.2]}
    RISK_DISTORTION_DICT_OPPONENT = {
        "pow":[ 1.0, 0.5, 0.2,-1.0,-0.5,-0.2,0.5,1.0,1.5,-0.5,-1.0,-1.5]}
    ##################################################
    start = time.time()
    MAX_NUM_JOB=20
    if EXPT_NAME == "train_iqn_vs_iqn":
        ray.init(logging_level=0, num_gpus=1, local_mode=False)
        srs=[]
        for env_id in ENV_IDS:
            for trial_idx, seed in enumerate(SEEDS):
                if trial_idx < 24: continue
                for risk_type, eta_list in RISK_DISTORTION_DICT.items():
                    for i in range(len(eta_list)):
                        eta=eta_list[i]
                        opponent_eta=RISK_DISTORTION_DICT_OPPONENT[risk_type][i]
    
                        srs.append(Actor.remote(env_id=env_id, 
                                        agent_learn_algo="iqn",
                                        opponent_learn_algo="iqn",
                                        eta=eta, opponent_eta=opponent_eta, risk_distortion=risk_type,
                                        seed=int(seed), trial_idx=trial_idx,
                                        ))

                        if len(srs)>=MAX_NUM_JOB:
                            jobs=[]
                            for sr in srs:
                                jobs.append(sr.train_iqn_iqn_indep.remote())
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
