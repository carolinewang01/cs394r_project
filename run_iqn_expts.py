import pprint


def train_iqn_vs_random(env_id, seed=1626, trial_idx=0):
    '''train iqn agent vs random
    '''
    from train_iqn_random import get_args, train_agent, watch

    args=get_args()
    args.env_id = env_id
    args.seed = seed
    args.trial_idx = trial_idx

    result, agent = train_agent(args)
    pprint.pprint(result)

def train_risk_aware(env_id, 
                     agent_learn_algo, opponent_resume_path,
                     cvar_eta=1.0,
                     seed=1626, trial_idx=0
    ):
    '''train args.algo agent vs pre-trained iqn agent

    Make sure to specify --agent-learn-algo, --cvar-eta, --opponent-resume-path
    '''
    from train_risk_aware import get_args, train_agent, watch

    args = get_args()
    args.env_id = env_id
    args.agent_learn_algo = agent_learn_algo
    args.opponent_resume_path = opponent_resume_path
    args.cvar_eta= cvar_eta
    args.seed = seed
    args.trial_idx = trial_idx

    result, agent = train_agent(args)
    pprint.pprint(result)

if __name__ == '__main__':
    # train single IQN vs random agent
    SEEDS = [1626, 
            174, 571, 2948, 109284
            ]
    ENV_IDS = ["leduc", 
               "tic-tac-toe", 
               "texas"
               ]
    CVAR_ETAS = [0.1, 
    0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    ]

    # train_iqn_vs_random(env_id="texas", seed=1626, trial_idx=0)

    for env_id in ENV_IDS:
        print("TRAINING IQN VS RANDOM")
        train_iqn_vs_random(env_id=env_id, seed=1626, trial_idx=0)
        for algo in ["iqn", "dqn"]:
            for trial_idx, seed in enumerate(SEEDS):
                for cvar_eta in CVAR_ETAS:
                    print(f"\nEXPT trial={trial_idx}, env_id={env_id}, algo={algo}, cvar_eta={cvar_eta}")
                    train_risk_aware(env_id=env_id,
                                     agent_learn_algo=algo,
                                     opponent_resume_path=f"log/{env_id}/iqn-vs-random_trial=0/policy.pth",
                                     cvar_eta=cvar_eta,
                                     seed=seed, trial_idx=trial_idx)