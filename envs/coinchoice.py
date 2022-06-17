import gym
from gym import spaces
import numpy as np
import random
import logging
from gym.envs.registration import register

class CoinChoiceEnv(gym.Env):
    def __init__(self):
        super(CoinChoiceEnv, self).__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-1.0,high=1.0,shape=(2,), dtype=np.float32)

    def step(self, action):
        reward = self.calReward(action)
        obs = np.random.uniform(-1,1,(2,)) # TODO: Can I put 0s here?
        done = True
        #print("action chosen:",action, " reward:",reward)
        return obs, reward, done, {}

    def calReward(self, action):
        rnd = np.random.uniform(0,1)
        if action == 0: #10% 9 90%0
            if rnd > 0.9 :
                return 9
            else:
                return 0
        elif action == 1: #90% 1 10%0
            if rnd > 0.1:
                return 1
            else:
                return 0
        return 0
            
    def reset(self):
        obs = np.random.uniform(-1,1,(2,))
        done = False
        return obs, done, {}

    def render(self):
        pass

gym.envs.register(
        id="CoinChoice-v0",
        entry_point="envs.coinchoice:CoinChoiceEnv",
        )

if __name__ == "__main__":
    #env = CoinChoiceEnv()
    env=gym.make("CoinChoice-v0")
    obs, done, info = env.reset()
    for i in range(10):
        action = random.choice([0,1])
        obs, reward, done, info = env.step(action)
        print("Coin: ", action, " Reward:",reward, "Obs:",obs)
        obs, done, info = env.reset()



