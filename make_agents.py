import torch
from tianshou.utils.net.common import Net
from tianshou.policy import (
    BasePolicy,
    DQNPolicy,
    RandomPolicy,
)

from models import RiskAwareIQN
from policies import RiskAwareIQNPolicy, CustomMAPolicyManager


def create_iqn_agent(args, cvar_eta):
    # feature_net = MLP(
    # *args.state_shape, args.action_shape, args.device, features_only=True
    # )
    net = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
    ).to(args.device)

    net = RiskAwareIQN(
        net,
        args.action_shape,
        args.hidden_sizes,
        num_cosines=args.num_cosines,
        cvar_eta=cvar_eta,
        device=args.device
    ).to(args.device)

    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    agent = RiskAwareIQNPolicy(
        net,
        optim,
        args.gamma,
        args.sample_size,
        args.online_sample_size,
        args.target_sample_size,
        args.n_step,
        target_update_freq=args.target_update_freq
    )
    return agent, optim

def create_dqn_agent(args):
    net = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
        # dueling=(Q_param, V_param),
    ).to(args.device)

    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    agent = DQNPolicy(
        net,
        optim,
        args.gamma,
        args.n_step,
        target_update_freq=args.target_update_freq
    )
    return agent, optim
