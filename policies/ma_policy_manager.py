from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from tianshou.data import Batch, ReplayBuffer
try:
    from tianshou.env.pettingzoo_env import PettingZooEnv
except ImportError:
    PettingZooEnv = None  # type: ignore
from tianshou.policy import MultiAgentPolicyManager, BasePolicy


class CustomMAPolicyManager(MultiAgentPolicyManager):
    '''
    Custom policy class allows turning off learning for one agent
    '''
    def __init__(
        self, policies: List[BasePolicy], env: PettingZooEnv, 
        policies_to_learn: dict,
        **kwargs: Any
    ) -> None:
        super().__init__(policies=policies, env=env, **kwargs)
        self.policies_to_learn = policies_to_learn

    def learn(self, batch: Batch,
              **kwargs: Any) -> Dict[str, Union[float, List[float]]]:
        """Dispatch the data to all policies for learning.

        :return: a dict with the following contents:

        ::

            {
                "agent_1/item1": item 1 of agent_1's policy.learn output
                "agent_1/item2": item 2 of agent_1's policy.learn output
                "agent_2/xxx": xxx
                ...
                "agent_n/xxx": xxx
            }
        """
        results = {}
        for agent_id, policy in self.policies.items():
            data = batch[agent_id]
            if not data.is_empty() and self.policies_to_learn[agent_id] == True:
                out = policy.learn(batch=data, **kwargs)
                for k, v in out.items():
                    results[agent_id + "/" + k] = v
        return results
