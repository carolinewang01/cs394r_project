import random
import numpy as np
from tianshou.data import Batch
from tianshou.policy import BasePolicy
from typing import Any, Dict, Optional, Union
try:
    from tianshou.env.pettingzoo_env import PettingZooEnv
except ImportError:
    PettingZooEnv = None  # type: ignore

class AgentPool(BasePolicy):
    def __init__(self,
                 env: PettingZooEnv,
                 max_len=10,
                 risk_aware=False,
                 **kwargs: Any):
        super().__init__(action_space=env.action_space, **kwargs)
        self.policies = []
        self.len = 0
        self.max_len = max_len
        self.eps = 0.0

        self.risk_aware = risk_aware

    def __len__(self):
        return self.len

    def add(self, policy):
        """
        Add a policy(tianshou.Policy class) to the agent pool
        Size of the agent pool increases
        """
        self.policies.append(policy)
        self.len = len(self.policies)
        # TODO size control

    def forward(
            self,
            batch: Batch,
            state: Optional[Union[dict, Batch, np.ndarray]] = None,
            **kwargs: Any,
            ) -> Batch:
        """
        Return a action picked by a joint policy
        If the size of the pool is greater than zero, return a uniformly sampled joint policy
        Else return a random policy
        """
        if self.len>0 and self.eps!=1.0:
            """
            Randomly select a policy and act
            """
            sampled_policy = self.policies[random.randint(0,len(self.policies)-1)]
            if self.risk_aware:
                sampled_policy.model.risk_distortion="wang"
                sampled_policy.model.eta = random.uniform(-0.2, 0.2) 
            return sampled_policy(batch, 
                                  state, 
                                  risk_aware=self.risk_aware,
                                  **kwargs)
        else:
            """
            Randomly Policy
            """
            mask = batch.obs.mask
            logits = np.random.rand(*mask.shape)
            logits[~mask] = -np.inf
            return Batch(act=logits.argmax(axis=-1)) 
     
    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        """Since an agent pool does not update, it returns an empty dict."""
        return {}
    
    def set_eps(self, eps):
        """
        TODO do something
        """
        self.eps = eps
        for p in self.policies:
            p.set_eps(eps)

    def sample(self):
        """
        Select and return policy from agent pool
        """
        index = random.randint(0, self.len)
        policy = self.policies[index]
        return policy

    def reset(self):
        """
        Clean up the agent pool
        """
        self.policies = []
        self.len = 0

