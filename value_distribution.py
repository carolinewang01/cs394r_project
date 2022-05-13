import gym
import numpy as np
import torch
from tianshou.data import Batch, to_numpy

import matplotlib.pyplot as plt
import seaborn
seaborn.set_theme(font_scale=2)

from make_agents import create_iqn_agent
from train_vs_random import get_env, get_args
from helpers import set_seed

ACTION_DICT = {0: "call", 
               1: "raise", 
               2: "fold", 
               3: "check"}


def plot_cdf(taus, logits, inverse=True, invalid_actions=[],
             eta=1.0, risk_distortion=None, 
             save=False, savename=None):
    for i in range(logits.shape[0]):
        if i in invalid_actions:
            continue
        logits_action = logits[i]
        if inverse:
            zipped = list(zip(taus, logits_action))
        else:
            zipped = list(zip(logits_action, taus))
            
        zipped.sort()
        sorted_xs, sorted_ys = list(zip(*zipped))

        plt.plot(sorted_xs, sorted_ys, label=ACTION_DICT[i])

    if inverse:
        xlabel, ylabel = "Taus", "Returns"
    else: 
        ylabel, xlabel = "Taus", "Returns"

    plt.legend()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title("Return Distribution")

    if save:
        plt.savefig(f"figures/return_distr_inverse={inverse}_eta={eta}_risk-distort={risk_distortion}_{savename}.pdf", bbox_inches="tight")

    plt.show()


def policy_forward_pass(env_id,
                        eta, risk_distortion, 
                        state_shape, action_shape,
                        obs):
    """
    obs should be a SINGLE observation

    Logits are = f(psi(x), phi(tau)) for all actions a
    Note that Q-values are obtained by averaging the logits

    ret.logits shape: (1, 4, 32) = (batch_size, num_actions, tau_sample_size)
    ret.tau shape: (1, 32) = (batch_size, tau_sample_size) 
    """
    # load in a sample policy
    args = get_args()
    args.env_id = env_id

    args.resume_path = f"log/{env_id}/iqn-vs-random_trial=0_eta={eta}_risk-distort={risk_distortion}/policy.pth"
    args.eta = eta
    args.risk_distortion = risk_distortion

    args.state_shape = state_shape
    args.action_shape = action_shape

    agent, _ = create_iqn_agent(args, eta=1.0, risk_distortion=None)
    agent.load_state_dict(torch.load(args.resume_path))
    # pass obs thru policy
    obs = np.expand_dims(obs, axis=0)
    batch = Batch(obs=obs, info=None)    
    agent.eval()
    ret = agent(batch) # returned batch structure: Batch(logits=logits, act=act, state=hidden, taus=taus)

    taus = to_numpy(ret.taus)[0]
    logits = to_numpy(ret.logits)[0]
    return taus, logits


def create_observation(private_card:str, community_card:str, community_card_known:bool, 
                       own_chips_raised:int, opponent_chips_raised:int):
    card_list = ["J", "Q", "K"]
    # checks to create valid observation
    assert (private_card in card_list) and (community_card in card_list)
    assert (own_chips_raised > 0) and (opponent_chips_raised > 0)

    observation = np.zeros(36)
    
    private_card_index = card_list.index(private_card)
    observation[private_card_index] = 1

    if community_card_known:
        community_card_index = card_list.index(community_card)
        observation[3 + community_card_index] = 1

    observation[3 + 3 + own_chips_raised] = 1
    observation[3 + 3 + 14 + opponent_chips_raised] = 1

    return observation



if __name__ == '__main__':
    ENV_ID = "leduc"
    ETA = 0.2
    RISK_DISTORTION = "cvar"

    env = get_env(ENV_ID)
    set_seed(seed=112358, envs=[env])
    observation_space = env.observation_space['observation'] if isinstance(
        env.observation_space, gym.spaces.Dict
    ) else env.observation_space
    state_shape = observation_space.shape or observation_space.n
    action_shape = env.action_space.shape or env.action_space.n

    # get sample observation
    # obs = env.reset()["obs"]

    savename = "round1"
    obs = create_observation(private_card="J", 
                             community_card="J", community_card_known=False,
                             own_chips_raised=2,
                             opponent_chips_raised=4
                            )
    invalid_actions = [3] # cannot check in this situation

    # savename = "round2"
    # obs = create_observation(private_card="J", 
    #                          community_card="J", community_card_known=True,
    #                          own_chips_raised=4,
    #                          opponent_chips_raised=4
    #                         )
    # invalid_actions = []

    taus, logits = policy_forward_pass(env_id=ENV_ID, eta=ETA, risk_distortion=RISK_DISTORTION,
                                       state_shape=state_shape, action_shape=action_shape,
                                       obs=obs
        )
    
    plot_cdf(taus, logits, inverse=True, invalid_actions=invalid_actions,
             eta=ETA, risk_distortion=RISK_DISTORTION, 
             save=True, savename=savename)
