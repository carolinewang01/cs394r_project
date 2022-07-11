from tianshou.env.pettingzoo_env import PettingZooEnv
from pettingzoo.classic import leduc_holdem_v4
env = PettingZooEnv(leduc_holdem_v4.env(num_players=2))

from IPython import embed; embed()
