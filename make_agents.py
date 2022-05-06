import torch
from tianshou.utils.net.common import Net
from tianshou.policy import (
    BasePolicy,
    DQNPolicy,
    RandomPolicy,
)

from models import RiskAwareIQN
from policies import RiskAwareIQNPolicy, CustomMAPolicyManager


def create_iqn_agent(args, eta, risk_distortion):
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
        eta=eta, # cvar_eta used at inference time only
        risk_distortion=risk_distortion,
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
