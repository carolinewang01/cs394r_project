import pprint
import time


def train_vs_random(env_id, 
                    agent_learn_algo,
                    eta=1.0,
                    risk_distortion=None,
                    seed=1626, 
                    trial_idx=0):
    '''train iqn agent vs random
    '''
    from train_vs_random import get_args, train_agent, watch

    args = get_args()
    args.env_id = env_id
    args.agent_learn_algo = agent_learn_algo
    args.eta = eta
    args.risk_distortion = risk_distortion
    args.seed = seed
    args.trial_idx = trial_idx

    result, agent = train_agent(args)
    pprint.pprint(result)

def train_risk_aware(env_id, 
                     agent_learn_algo, 
                     opponent_learn_algo, 
                     opponent_resume_path,
                     eta=1.0, 
                     risk_distortion=None,
                     seed=1626, 
                     trial_idx=0
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

if __name__ == '__main__':
    start = time.time()
    SEEDS = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9469, 5293, 1436, 564, 3654, 5106, 8570, 5858, 9603, 4703, #1626, 174, 571, 2948, 109284
            ]
    ENV_IDS = [
               "leduc", 
               # "texas",
               # "tic-tac-toe",  # order of agents fixed, need to fix this
               # "texas-no-limit" # order of agents fixed, need to fix this
               ]
    OPPONENT_LEARN_ALGO = ["iqn", "dqn"]
    AGENT_LEARN_ALGO = ["iqn", "dqn"]

    RISK_DISTORTION_DICT = { # possible eta values
        "cvar": [0.2, 0.4, 0.6, 0.8, 1.0],
        "wang": [-0.75, -0.25, 0.0, 0.25, 0.75], # positive corresponds to risk seeking, negative to risk averse
        "pow": [-2.5, -1.5, 0.0, 1.5, 2.5] # positive corresponds to risk seeking, negative to risk averse
    }

    for env_id in ENV_IDS:
        for seed in SEEDS:
            trial_idx = seed
            for algo in AGENT_LEARN_ALGO:
                if algo=='dqn':
                    train_vs_random(env_id=env_id,
                                         agent_learn_algo=algo,
                                         eta=1.0, # doesn't matter what this is
                                         risk_distortion=None,
                                         seed=seed,
                                         trial_idx=trial_idx
                                        )
                if algo=='iqn':
                    risk_distortion='pow'
                    for eta in RISK_DISTORTION_DICT[risk_distortion]:
                        train_vs_random(env_id=env_id,
                                        agent_learn_algo=algo,
                                        eta=eta,
                                        risk_distortion=risk_distortion,
                                        seed=seed,
                                        trial_idx=trial_idx
                                        )
    '''
    RISK_TYPE = "cvar"
    # train_risk_aware(env_id="leduc",
    #                  agent_learn_algo="dqn",
    #                  opponent_learn_algo="iqn",
    #                  opponent_resume_path=f"log/leduc/iqn-vs-random_trial=0/policy.pth",
    #                  eta=-2, risk_distortion="pow",
    #                  seed=10000, trial_idx=0)

    # import sys; sys.exit(0)

    for env_id in ENV_IDS:
        for algo_v_random in OPPONENT_LEARN_ALGO:
            print(f"TRAINING {algo_v_random} OPPONENT")
            train_vs_random(env_id=env_id, 
                            agent_learn_algo=algo_v_random, 
                            eta=1.0, risk_distortion=None,
                            seed=1626, trial_idx=0)

            for algo in AGENT_LEARN_ALGO:
                for trial_idx, seed in enumerate(SEEDS):
                    if algo_v_random == "iqn":
                        for eta in RISK_DISTORTION_DICT[RISK_TYPE]:
                            print(f"\nEXPT trial={trial_idx}, env_id={env_id}, agent algo={algo}, opponent algo={algo_v_random}, eta={eta}")
                            train_risk_aware(env_id=env_id,
                                             agent_learn_algo=algo,
                                             opponent_learn_algo=algo_v_random,
                                             opponent_resume_path=f"log/{env_id}/{algo_v_random}-vs-random_trial=0/policy.pth",
                                             eta=eta, risk_distortion=RISK_TYPE,
                                             seed=seed, trial_idx=trial_idx)
                    elif algo_v_random == "dqn": 
                        print(f"\nEXPT trial={trial_idx}, env_id={env_id}, agent algo={algo}, opponent algo={algo_v_random}")
                        train_risk_aware(env_id=env_id,
                                         agent_learn_algo=algo,
                                         opponent_learn_algo=algo_v_random,
                                         opponent_resume_path=f"log/{env_id}/{algo_v_random}-vs-random_trial=0/policy.pth",
                                         eta=1.0, risk_distortion=None,# doesn't matter what this is
                                         seed=seed, trial_idx=trial_idx)
    '''
    end = time.time()
    print("SCRIPT RUN TIME: ", end - start)
