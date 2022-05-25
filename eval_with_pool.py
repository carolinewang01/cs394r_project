import pprint
import time
import glob
import numpy as np
from datetime import timedelta

def test_pool(
            env_id='leduc',
            agent_learn_algo='iqn',
            agent_resume_path=None,
            opponent_algo='iqn',
            opponent_resume_path=None,
            ):
    
    # from train_vs_random import get_args, watch
    from train_risk_aware import get_args, watch

    args = get_args()
    
    args.env_id = env_id
    args.agent_learn_algo = agent_learn_algo
    args.agent_resume_path = agent_resume_path
    args.opponent_learn_algo = opponent_algo
    args.opponent_resume_path = opponent_resume_path

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
    
    end = time.time()
    elapsed = str(timedelta(seconds=end - start))
    print("SCRIPT RUN TIME: ", elapsed)
