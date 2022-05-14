import pprint
import time
import numpy as np
from datetime import timedelta


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

def test_sp():
    '''
    env_id,
            agent_learn_algo,
            agent_resume_path,
            opponent_algo,
            opponent_resume_path,
            ):
    '''
    from train_iqn_sp import get_args, watch
    args = get_args()
    '''
    args.env_id = env_id
    args.agent_learn_algo = agent_learn_algo
    args.agent_resume_path = agent_resume_path
    args.opponent_algo = opponent_algo
    args.opponent_resume_path = opponent_resume_path
    '''
    watch(args)


if __name__ == '__main__':
    SEEDS = [1626, 174, 571, 2948, 109284]
    SEEDS = np.load('seeds.npy')
    ENV_IDS = ["leduc", 
               "texas",
               # "texas-no-limit" # order of agents fixed, need to fix this
               ]
    
    EXPT_NAME = "train_sp" #"train_sp_risk_aware" # "train_sp"
    RISK_AWARE = [True, False]
    ##################################################
    start = time.time()
    
    if EXPT_NAME == "train_sp":
        for env_id in ENV_IDS:
            for trial_idx, seed in enumerate(SEEDS):
                for risk_aware in RISK_AWARE:
                        train_sp(env_id=env_id, 
                                        agent_learn_algo="iqn",
                                        eta=1.0, risk_distortion=None,
                                        seed=int(seed), trial_idx=trial_idx,
                                        risk_aware=risk_aware)
    elif EXPT_NAME == "test_sp":
        test_sp()
    end = time.time()
    elapsed = str(timedelta(seconds=end - start))
    print("SCRIPT RUN TIME: ", elapsed)
