import pprint
import time


def train_vs_random(env_id, 
                    agent_learn_algo, 
                    seed=1626, 
                    cvar_eta=1.0,
                    trial_idx=0):
    '''train iqn agent vs random
    '''
    from train_vs_random import get_args, train_agent, watch

    args = get_args()
    args.env_id = env_id
    args.agent_learn_algo = agent_learn_algo
    args.cvar_eta = cvar_eta
    args.seed = seed
    args.trial_idx = trial_idx

    result, agent = train_agent(args)
    pprint.pprint(result)

def train_risk_aware(env_id, 
                     agent_learn_algo, 
                     opponent_learn_algo, opponent_resume_path,
                     cvar_eta=1.0,
                     seed=1626, 
                     trial_idx=0
    ):
    '''train args.algo agent vs pre-trained iqn agent

    Make sure to specify --agent-learn-algo, --cvar-eta, --opponent-resume-path
    '''
    from train_risk_aware import get_args, train_agent, watch

    args = get_args()
    args.env_id = env_id
    args.agent_learn_algo = agent_learn_algo
    args.opponent_learn_algo = opponent_learn_algo

    args.opponent_resume_path = opponent_resume_path
    args.cvar_eta= cvar_eta
    args.seed = seed
    args.trial_idx = trial_idx

    result, agent = train_agent(args)
    pprint.pprint(result)

if __name__ == '__main__':
    start = time.time()
    SEEDS = [
            1626, #174, 571, 2948, 109284
            ]
    ENV_IDS = [
               "leduc", 
               # "texas",
               # "tic-tac-toe",  # order of agents fixed, need to fix this
               # "texas-no-limit" # order of agents fixed, need to fix this
               ]
    OPPONENT_LEARN_ALGO = ["iqn", "dqn"]
    AGENT_LEARN_ALGO = ["iqn", "dqn"]

    CVAR_ETAS = [0.2, 0.4, 0.6, 0.8, 1.0
    ]

    for env_id in ENV_IDS:
        for algo in AGENT_LEARN_ALGO:
            if algo=='dqn':
                        train_vs_random(env_id=env_id,
                                         agent_learn_algo=algo,
                                         cvar_eta=1.0, # doesn't matter what this is
                                        )
            if algo=='iqn':
                    for cvar_eta in CVAR_ETAS:
                        train_vs_random(env_id=env_id,
                                        agent_learn_algo=algo,
                                        cvar_eta=cvar_eta)
    '''
    for env_id in ENV_IDS:
        for algo_v_random in OPPONENT_LEARN_ALGO:
            print(f"TRAINING {algo_v_random} OPPONENT")
            train_vs_random(env_id=env_id, agent_learn_algo=algo_v_random, seed=1626, trial_idx=0)

            for algo in AGENT_LEARN_ALGO:
                for trial_idx, seed in enumerate(SEEDS):
                    if algo_v_random == "iqn":
                        for cvar_eta in CVAR_ETAS:
                            print(f"\nEXPT trial={trial_idx}, env_id={env_id}, agent algo={algo}, opponent algo={algo_v_random}, cvar_eta={cvar_eta}")
                            train_risk_aware(env_id=env_id,
                                             agent_learn_algo=algo,
                                             opponent_learn_algo=algo_v_random,
                                             opponent_resume_path=f"log/{env_id}/{algo_v_random}-vs-random_trial=0/policy.pth",
                                             cvar_eta=cvar_eta,
                                             seed=seed, trial_idx=trial_idx)
                    elif algo_v_random == "dqn": 
                        print(f"\nEXPT trial={trial_idx}, env_id={env_id}, agent algo={algo}, opponent algo={algo_v_random}")
                        train_risk_aware(env_id=env_id,
                                         agent_learn_algo=algo,
                                         opponent_learn_algo=algo_v_random,
                                         opponent_resume_path=f"log/{env_id}/{algo_v_random}-vs-random_trial=0/policy.pth",
                                         cvar_eta=1.0, # doesn't matter what this is
                                         seed=seed, trial_idx=trial_idx)
    '''
    end = time.time()
    print("SCRIPT RUN TIME: ", end - start)
