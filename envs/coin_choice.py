import gym
from gym import spaces
import numpy as np
import random

class CoinChoiceEnv(gym.Env):
    def __init__(self):
        super(CoinChoiceEnv, self).__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(2)

    def step(self, action):
        reward = self.calReward(action)
        obs = np.random.uniform(-1,1,2) # TODO: Can I put 0s here?
        done = True
        return obs, reward, done, {}

    def calReward(self, action):
        rnd = np.random.uniform(0,1)
        if action == 0:
            if rnd > 0.9 :
                return 9
            else:
                return 0
        elif action == 1:
            if rnd > 0.1:
                return 1
            else:
                return 0
        return 0
            
    def reset(self):
        obs = np.random.uniform(-1,1,2)
        done = False
        return obs, done, {}

    def render(self):
        pass

gym.envs.register(
        id="coinchoice-v0",
        entry_point="envs.coin_choice:CoinChoiceEnv",
        )

if __name__ == "__main__":
    #env = CoinChoiceEnv()
    env=gym.make("coinchoice-v0")
    obs, done, info = env.reset()
    for i in range(10):
        action = random.choice([0,1])
        obs, reward, done, info = env.step(action)
        print("Coin: ", action, " Reward:",reward)
        obs, done, info = env.reset()



