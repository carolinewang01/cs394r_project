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
    
    from train_iqn_sp import get_args, watch
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
    agent_resume_path = 'log/selfplay/leduc/iqn-selfplay_trial=0_riskaware=False/9/policy.pth'
    opponents_pool_path = 'log/leduc-pool/'
    rew, winrate, wintierate = None, None, None
    for opponent in glob.glob(opponents_pool_path+'iqn-vs-random_trial=*=pow'):
        opponent_resume_path = opponent+'/policy.pth'
        print('Main Agent:', agent_resume_path)
        print('Opponent:',opponent_resume_path)
        try:
            rew, winrate,wintierate = test_pool(
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
            end = time.time()
            elapsed = str(timedelta(seconds=end - start))
            print("SCRIPT RUN TIME: ", elapsed)
            print('mean reward:',np.mean(rews), ' std:', np.std(rews), ' winrate:', np.mean(winrates)*100, ' wintie rate:', np.mean(wintierates)*100, ' num of seeds:', len(rews))
    
    end = time.time()
    elapsed = str(timedelta(seconds=end - start))
    print("SCRIPT RUN TIME: ", elapsed)
