import random
import numpy as np
import gym
from gym.spaces import Discrete, Box, Dict

class SimpleBetEnv(gym.Env):
    def __init__(self):
        super(SimpleBetEnv, self).__init__()

        self.agents = ["player_0", "player_1"]
        self.agent_idx = {'player_0': 0, 'player_1': 1}
        self.n_actions = 2 # 0 is fold, 1 is bet
        self.n_obs =  3 # own card
        self.action_spaces = {"player_0": Discrete(self.n_actions),
                              "player_1": Discrete(self.n_actions)
        }

        '''Observation space should include
        - players own card (onehot)
        - amt contributed by self to pot (onehot)
        - amt contributed by opponent to pot (onehot)
        '''
        self.observation_spaces = {"player_0": Dict({
                                                # "agent_id": Discrete(2),
                                                "action_mask": Box(0, 1, (self.n_actions,), int), 
                                                "observation": Box(0, 1, (self.n_obs,), int)
                                                }),
                                    "player_1": Dict({
                                                # "agent_id": Discrete(2),
                                                "action_mask": Box(0, 1, (self.n_actions,), int), 
                                                "observation": Box(0, 1, (self.n_obs,), int)
                                                })
        }
        # dummy action space
        self.action_space = Discrete(self.n_actions)  # not sure these are needed?
        self.observation_space = Dict({
                                       # "agent_id": Discrete(2),
                                       "action_mask": Box(0, 1, (self.n_actions,), int), 
                                       "observation": Box(0, 1, (self.n_obs,), int)
                                       })

        self.deck = ["A", "K", "Q"] 
        self.card_rank = ["Q", "K", "A"] # Q < K < A
        self.reset()

    def switch_player(self):
        current_player_idx = self.agents.index(self.game_state["current_player"])
        self.game_state["current_player"] = self.agents[1 - current_player_idx]

    def reset(self):
        # select starting player
        play_order = random.sample(self.agents, len(self.agents))
        # distribute cards
        random.shuffle(self.deck) 
        self.game_state = {"player_0": {"card": self.deck[0], "action": None}, 
                           "player_1": {"card": self.deck[1], "action": None}, 
                           "community": self.deck[2],
                           "play_order": play_order,
                           "current_player": play_order[0]}
        obs = self.get_obs()
        return obs

    def step(self, action):
        """Should return something of form
        (
        {'agent_id': 'player_1', 
            'obs': array([1], dtype=int),   # observation
            'mask': [True, True, True, False]}, 
       [0, 0],                              # reward for both agents
       False,                               # whether or not game is done
       {'legal_moves': []}                  # whether or not current move is legal
       )
        """
        self.game_state[self.game_state["current_player"]]["action"] = action
        rews = self.compute_reward(action)

        self.switch_player()
        obs = self.get_obs()

        starting_player = self.game_state["play_order"][0]
        done = (self.game_state["current_player"] == starting_player)

        # self.switch_player()
        return obs, rews, done, {}

    def get_obs(self):
        '''Should return something of form
        {
        'agent_id': 'player_0', 
        'obs': array([1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0.], dtype=float32), 
       'mask': [True, True, True, False]}
        '''
        current_player = self.game_state["current_player"]
        player_card = self.game_state[current_player]["card"]
        obs = np.array([int(card == player_card) for card in self.card_rank]) # onehot observation
        action_mask = [True, True] # all actions always available 

        return {"agent_id": current_player,
                "obs": obs,
                "mask": action_mask
                }

    def compute_reward(self, action):
        starting_player = self.game_state["play_order"][0]
        # both moves aren't in yet
        if self.game_state["player_0"]["action"] is None or self.game_state["player_1"]["action"] is None:
            return [0, 0]
        # both players folded
        elif self.game_state["player_0"]["action"] == 0 and  self.game_state["player_1"]["action"] == 0:
            return [0, 0]
        # player_0 folded, player_1 bet
        elif self.game_state["player_0"]["action"] == 0 and self.game_state["player_1"]["action"] == 1:
            return [0, 1]
        # player_1 bet, player_0 folded
        elif self.game_state["player_0"]["action"] == 1 and self.game_state["player_1"]["action"] == 0:
            return [1, 0]
        # both bet
        player0_rank = self.card_rank.index(self.game_state["player_0"]["card"])
        player1_rank = self.card_rank.index(self.game_state["player_1"]["card"])
        if player0_rank > player1_rank:
            return [3, -3]
        else:
            return [-3, 3]

    def render(self):
        pass

gym.envs.register(
        id="simplebet-v0",
        entry_point="envs.simple_bet:SimpleBetEnv",
        )

if __name__ == "__main__":
    env = gym.make("simplebet-v0")
    obs = env.reset()
    done = False
    print("OBS IS ", obs)
    while not done:
        action = random.choice([0,1])
        print("ACTION IS ", action)
        obs, reward, done, info = env.step(action)
        print("OBS IS" , obs)
        print("REWARD IS ", reward)
        print("DONE IS ", done)
    print(env.game_state)




