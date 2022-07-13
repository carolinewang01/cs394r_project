import pprint
import time
import numpy as np
from datetime import timedelta
import ray

def train_vs_random(env_id, agent_learn_algo, 
                    eta=1.0, risk_distortion=None,
                    seed=1626, trial_idx=0):
    '''train iqn agent vs random
    '''
    from train_vs_random import get_args, train_agent, watch

    args=get_args()
    args.agent_learn_algo = agent_learn_algo
    args.env_id = env_id
    args.eta = eta
    args.risk_distortion = risk_distortion
    args.seed = seed
    args.trial_idx = trial_idx

    result, agent = train_agent(args)
    pprint.pprint(result)

def train_risk_aware(env_id, 
                     agent_learn_algo, 
                     opponent_learn_algo, opponent_resume_path,
                     eta=1.0, risk_distortion=None,
                     seed=1626, trial_idx=0
    ):
    '''train args.algo agent vs pre-trained iqn agent

    Make sure to specify --agent-learn-algo, --eta, --risk-distortion, --opponent-resume-path
    '''
    from train_risk_aware import get_args, train_agent, watch

    args = get_args()
    args.env_id = env_id
    args.agent_learn_algo = agent_learn_algo
    args.opponent_learn_algo = opponent_learn_algo

    args.opponent_resume_path = opponent_resume_path
    args.eta = eta
    args.risk_distortion = risk_distortion
    args.seed = seed
    args.trial_idx = trial_idx

    result, agent = train_agent(args)
    pprint.pprint(result)

def train_iqn_dqn_indep(env_id,
                        agent_learn_algo,
                        opponent_learn_algo,
                        opponent_resume_path,
                        eta=1.0, risk_distortion=None,
                        seed=1626, trial_idx=0):
    from train_iqn_vs_dqn import get_args, train_agent, watch
    args = get_args()
    args.env_id = env_id
    args.agent_learn_algo = agent_learn_algo
    args.opponent_learn_algo = opponent_learn_algo

    args.opponent_resume_path = opponent_resume_path
    args.eta = eta
    args.risk_distortion = risk_distortion
    args.seed = seed
    args.trial_idx = trial_idx

    result, agent = train_agent(args)
    pprint.pprint(result)

def train_iqn_iqn_indep(env_id,
                        agent_learn_algo,
                        opponent_learn_algo,
                        opponent_resume_path,
                        eta=1.0, opponent_eta=1.0, risk_distortion=None,
                        seed=1626, trial_idx=0):
    from train_iqn_vs_iqn import get_args, train_agent, watch
    args = get_args()
    args.env_id = env_id
    args.agent_learn_algo = agent_learn_algo
    args.opponent_learn_algo = opponent_learn_algo

    args.opponent_resume_path = opponent_resume_path
    args.eta = eta
    args.opponent_eta = opponent_eta
    args.risk_distortion = risk_distortion
    args.seed = seed
    args.trial_idx = trial_idx

    result, agent = train_agent(args)
    pprint.pprint(result)


if __name__ == '__main__':
    SEEDS = [1626, 174, 571, 2948, 109284]
    ENV_IDS = ["leduc", 
               "texas",
               # "texas-no-limit" # order of agents fixed, need to fix this
               ]
    RISK_DISTORTION_DICT = { # possible eta values
        "cvar": [0.2, 0.4, 0.6, 0.8, 1.0],
        "wang": [-0.75, -0.25, 0.25, 0.75], # positive corresponds to risk seeking, negative to risk averse
        "pow": [-2.5, -1.5, 1.5, 2.5] # positive corresponds to risk seeking, negative to risk averse
    }
    
    EXPT_NAME = "train_iqn_vs_iqn" #"train_vs_risk_aware" # "train_vs_random"
    ##################################################
    start = time.time()

    if EXPT_NAME == "train_vs_random":
        for env_id in ENV_IDS:
            for trial_idx, seed in enumerate(SEEDS):
                # train dqn v random
                train_vs_random(env_id=env_id, 
                                agent_learn_algo="dqn", 
                                eta=1.0, risk_distortion=None,
                                seed=seed, trial_idx=trial_idx)
                # train iqn v random for all risk types
                for risk_type, eta_list in RISK_DISTORTION_DICT.items():
                    for eta in eta_list:
                        train_vs_random(env_id=env_id, 
                                        agent_learn_algo="iqn", 
                                        eta=eta, risk_distortion=risk_type,
                                        seed=seed, trial_idx=trial_idx)

    elif EXPT_NAME == "train_vs_risk_aware":
        OPPONENT_LEARN_ALGO = ["iqn", "dqn"]
        AGENT_LEARN_ALGO = ["iqn", "dqn"]

        for env_id in ENV_IDS:
            for trial_idx, seed in enumerate(SEEDS):
                for opponent_algo in OPPONENT_LEARN_ALGO:
                    for algo in AGENT_LEARN_ALGO:

                        if opponent_algo == "iqn":
                            for risk_type, eta_list in RISK_DISTORTION_DICT.items():
                                for eta in eta_list:
                                    train_risk_aware(env_id=env_id,
                                                     agent_learn_algo=algo,
                                                     opponent_learn_algo=opponent_algo,
                                                     opponent_resume_path=f"log/{env_id}/{opponent_algo}-vs-random_trial=0_eta={eta}_risk-distort={risk_type}/policy.pth",
                                                     eta=eta, risk_distortion=risk_type,
                                                     seed=seed, trial_idx=trial_idx)
                        elif opponent_algo == "dqn": 
                            train_risk_aware(env_id=env_id,
                                             agent_learn_algo=algo,
                                             opponent_learn_algo=opponent_algo,
                                             opponent_resume_path=f"log/{env_id}/{opponent_algo}-vs-random_trial=0_eta=1.0_risk-distort=None/policy.pth",
                                             eta=1.0, risk_distortion=None,# doesn't matter what this is
                                             seed=seed, trial_idx=trial_idx)

    elif EXPT_NAME == "train_iqn_vs_dqn":
        OPPONENT_LEARN_ALGO = ["dqn"]
        AGENT_LEARN_ALGO = ["iqn"]
        RISK_DISTORTION_DICT = {"pow":[-1.0,-0.5,-0.2,0,0.2,0.5,1.0]}
        SEEDS=np.load('seeds.npy')
        for env_id in ENV_IDS:
            for trial_idx, seed in enumerate(list(SEEDS)):
                for opponent_algo in OPPONENT_LEARN_ALGO:
                    for algo in AGENT_LEARN_ALGO:
                            for risk_type, eta_list in RISK_DISTORTION_DICT.items():
                                for eta in eta_list:
                                    train_iqn_dqn_indep(env_id=env_id,
                                                     agent_learn_algo=algo,
                                                     opponent_learn_algo=opponent_algo,
                                                     opponent_resume_path=None,
                                                     eta=eta, risk_distortion=risk_type,
                                                     seed=int(seed), trial_idx=trial_idx)
    elif EXPT_NAME == "train_iqn_vs_iqn":
        OPPONENT_LEARN_ALGO = ["iqn"]
        AGENT_LEARN_ALGO = ["iqn"]
        SEEDS=np.load('seeds.npy')
        RISK_DISTORTION_DICT = {
                "pow":[-0.2,-0.2,-0.2, 0.2, 0.2, 0.2,0.2,0.2,0.2,-0.2,-0.2,-0.2]}
        RISK_DISTORTION_DICT_OPPONENT = {
                "pow":[ 1.0, 0.5, 0.2,-1.0,-0.5,-0.2,0.5,1.0,1.5,-0.5,-1.0,-1.5]}
        for env_id in ENV_IDS:
            for trial_idx, seed in enumerate(SEEDS):
                for opponent_algo in OPPONENT_LEARN_ALGO:
                    for algo in AGENT_LEARN_ALGO:
                            for risk_type, eta_list in RISK_DISTORTION_DICT.items():
                                for i in range(len(eta_list)):
                                    eta=eta_list[i]
                                    opponent_eta=RISK_DISTORTION_DICT_OPPONENT[risk_type][i]
                                    train_iqn_iqn_indep(env_id=env_id,
                                                     agent_learn_algo=algo,
                                                     opponent_learn_algo=opponent_algo,
                                                     opponent_resume_path=None,
                                                     eta=eta, opponent_eta=opponent_eta, risk_distortion=risk_type,
                                                     seed=int(seed), trial_idx=trial_idx)
  
    end = time.time()
    elapsed = str(timedelta(seconds=end - start))
    print("SCRIPT RUN TIME: ", elapsed)
