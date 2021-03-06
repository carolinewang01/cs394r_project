#!/scratch/cluster/clw4542/conda_envs/cs394r/bin/python

import argparse
import os
from copy import deepcopy
from typing import Optional, Tuple

import gym
import numpy as np
import torch
from pettingzoo.classic import leduc_holdem_v4, tictactoe_v3, texas_holdem_v4, texas_holdem_no_limit_v6
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import (
    BasePolicy,
    DQNPolicy,
    MultiAgentPolicyManager,
    RandomPolicy,
)
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger

# ours
from envs.simple_bet import SimpleBetEnv
from make_agents import create_iqn_agent, create_dqn_agent
from helpers import set_seed


def get_env(env_id):
    if env_id == "leduc":
        env = PettingZooEnv(leduc_holdem_v4.env(num_players=2))
    elif env_id == "tic-tac-toe":
        env = PettingZooEnv(tictactoe_v3.env())
    elif env_id == "texas":
        env = PettingZooEnv(texas_holdem_v4.env(num_players=2))
    elif env_id == "texas-no-limit":
        env = PettingZooEnv(texas_holdem_no_limit_v6.env(num_players=2))
    elif env_id == "simple-bet":
        env = SimpleBetEnv()
    else:
        raise Exception(f"Env name {env_id} not valid.")
    return env


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env-id', type=str, choices=["leduc", "tic-tac-toe", "texas", "simple-bet"], default=None)
    parser.add_argument('--agent-learn-algo', type=str, choices=[
                        "dqn", "iqn"], default=None, help='algorithm for the agent to learn with')

    parser.add_argument('--eta', type=float, default=1.0,
                        help='eta param of opponent IQN agent')
    parser.add_argument('--risk-distortion', choices=[
                        "cvar", "wang", "pow", None, "None"], help='distortion type of opponent IQN agent')

    ############################
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--trial-idx', type=int, default=0)
    parser.add_argument('--eps-test', type=float, default=0.)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument(
        '--gamma', type=float, default=0.9, help='a smaller gamma favors earlier win'
    )

    # IQN args
    parser.add_argument('--sample-size', type=int, default=32)
    parser.add_argument('--online-sample-size', type=int, default=8)
    parser.add_argument('--target-sample-size', type=int, default=8)
    parser.add_argument('--num-cosines', type=int, default=64)
    # IQN/DQN args
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=320)

    # usual RL args
    parser.add_argument('--epoch', type=int, default=100)
    # number transitions collected per epoch
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    # number transitions per call of collector.collect()
    parser.add_argument('--step-per-collect', type=int, default=100)
    # policy will be updated update-per-step*step-per-collect times per call of collector.collect()
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument(
        '--hidden-sizes', type=int, nargs='*', default=[64, 64]
    )
    parser.add_argument('--num-training-envs', type=int, default=10)
    parser.add_argument('--num-test-envs', type=int, default=5)
    parser.add_argument('--episode-per-test', type=int, default=50)

    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.1)
    parser.add_argument(
        '--win-rate',
        type=float,
        default=0.7,
        help='the expected winning rate: Optimal policy can get 0.7'
    )
    parser.add_argument(
        '--watch',
        default=False,
        action='store_true',
        help='no training, '
        'watch the play of pre-trained models'
    )
    parser.add_argument(
        '--agent-id',
        type=int,
        default=2,
        help='the learned agent plays as the'
        ' agent_id-th player. Choices are 1 and 2.'
    )
    parser.add_argument(
        '--resume-path',
        type=str,
        default='',
        help='the path of agent pth file '
        'for resuming from a pre-trained agent'
    )
    parser.add_argument(
        '--opponent-resume-path',
        type=str,
        default='',
        help='the path of opponent agent pth file '
        'for resuming from a pre-trained agent'
    )
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]


def get_agents(
    args: argparse.Namespace = get_args(),
    agent_learn: Optional[BasePolicy] = None,
    agent_opponent: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = get_env(args.env_id)
    observation_space = env.observation_space['observation'] if isinstance(
        env.observation_space, gym.spaces.Dict
    ) else env.observation_space

    args.state_shape = observation_space.shape or observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # while len(args.state_shape) < 3:
    # args.state_shape += (1,)

    if agent_learn is None:
        # define model
        if args.agent_learn_algo == "iqn":
            agent_learn, optim = create_iqn_agent(
                args, eta=args.eta, risk_distortion=args.risk_distortion)
        elif args.agent_learn_algo == "dqn":
            agent_learn, optim = create_dqn_agent(args)

        if args.resume_path:
            agent_learn.load_state_dict(torch.load(args.resume_path))

    if agent_opponent is None:
        if args.opponent_resume_path:
            agent_opponent, _ = create_iqn_agent(
                args, eta=args.eta, risk_distortion=args.risk_distortion)  # deepcopy(agent_learn)
            agent_opponent.load_state_dict(
                torch.load(args.opponent_resume_path))
        else:
            agent_opponent = RandomPolicy()

    if args.agent_id == 1:
        agents = [agent_learn, agent_opponent]
    else:
        agents = [agent_opponent, agent_learn]
    policy = MultiAgentPolicyManager(agents, env)
    return policy, optim, env.agents


def train_agent(
    args: argparse.Namespace = get_args(),
    agent_learn: Optional[BasePolicy] = None,
    agent_opponent: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[dict, BasePolicy]:

    def env_fn(): return get_env(args.env_id)
    train_envs = SubprocVectorEnv(
        [env_fn for _ in range(args.num_training_envs)])
    test_envs = SubprocVectorEnv([env_fn for _ in range(args.num_test_envs)])

    set_seed(args.seed, envs=[train_envs, test_envs])

    policy, optim, agents = get_agents(
        args, agent_learn=agent_learn, agent_opponent=agent_opponent, optim=optim
    )

    # collector
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    train_collector.collect(n_step=args.batch_size * args.num_training_envs)
    # log

    log_path = os.path.join(args.logdir, args.env_id,
                            f'{args.agent_learn_algo}-vs-random_trial={args.trial_idx}_eta={args.eta}_risk-distort={args.risk_distortion}')
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        if hasattr(args, 'model_save_path'):
            model_save_path = args.model_save_path
        else:
            model_save_path = os.path.join(
                args.logdir, args.env_id, f'{args.agent_learn_algo}-vs-random_trial={args.trial_idx}_eta={args.eta}_risk-distort={args.risk_distortion}', 'policy.pth'
            )
        torch.save(
            policy.policies[agents[args.agent_id - 1]
                            ].state_dict(), model_save_path
        )

    def stop_fn(mean_rewards):
        return None  # mean_rewards >= args.win_rate

    def train_fn(epoch, env_step):
        policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_train)

    def test_fn(epoch, env_step):
        policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_test)

    def reward_metric(rews):
        return rews[:, args.agent_id - 1]

    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.episode_per_test,
        batch_size=args.batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=args.update_per_step,
        logger=logger,
        test_in_train=True,
        reward_metric=reward_metric
    )

    return result, policy.policies[agents[args.agent_id - 1]]


def watch(
    args: argparse.Namespace = get_args(),
    agent_learn: Optional[BasePolicy] = None,
    agent_opponent: Optional[BasePolicy] = None,
) -> None:
    def env_fn(): return get_env(args.env_id)

    env = SubprocVectorEnv([env_fn for _ in range(args.num_test_envs)])
    policy, optim, agents = get_agents(
        args, agent_learn=agent_learn, agent_opponent=agent_opponent
    )
    policy.eval()
    #policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_test)
    for i in range(2):
        policy.policies[agents[i]].set_eps(0.0)
    collector = Collector(policy, env, exploration_noise=True)
    result = collector.collect(n_episode=args.episode_per_test, render=False)
    rews, lens = result["rews"], result["lens"]
    print(
        f"Final reward: {rews[:, args.agent_id - 1].mean()}, length: {lens.mean()}")
    return np.mean(rews[:, args.agent_id - 1]), np.mean(rews[:, args.agent_id - 1] > 0), np.mean(rews[:, args.agent_id - 1] >= 0)

# def watch(
#     args: argparse.Namespace = get_args(),
#     agent_learn: Optional[BasePolicy] = None,
#     agent_opponent: Optional[BasePolicy] = None,
# ) -> None:
#     env_fn = lambda: get_env(args.env_id)
#     env = DummyVectorEnv([env_fn])
#     policy, optim, agents = get_agents(
#         args, agent_learn=agent_learn, agent_opponent=agent_opponent
#     )
#     policy.eval()
#     policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_test)
#     collector = Collector(policy, env, exploration_noise=True)
#     result = collector.collect(n_episode=1, render=args.render)
#     rews, lens = result["rews"], result["lens"]
#     print(f"Final reward: {rews[:, args.agent_id - 1].mean()}, length: {lens.mean()}")


if __name__ == '__main__':
    args = get_args()
    result, agent = train_agent(args)
