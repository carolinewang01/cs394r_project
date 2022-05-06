from typing import Any, Dict, Optional, Union
from tianshou.data import Batch, to_numpy

import numpy as np
import torch

from tianshou.policy import IQNPolicy

class RiskAwareIQNPolicy(IQNPolicy):
    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        discount_factor: float = 0.99,
        sample_size: int = 32,
        online_sample_size: int = 8,
        target_sample_size: int = 8,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        **kwargs: Any,
    ) -> None:

        super().__init__(model=model, optim=optim, 
            discount_factor=discount_factor, 
            sample_size=sample_size,
            online_sample_size=online_sample_size,
            target_sample_size=target_sample_size,
            estimation_step=estimation_step,
            target_update_freq=target_update_freq,
            reward_normalization=reward_normalization,
            **kwargs)

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        if model == "model_old":
            sample_size = self._target_sample_size
        elif self.training:
            sample_size = self._online_sample_size
        else:
            sample_size = self._sample_size
        model = getattr(self, model)
        obs = batch[input]
        obs_next = obs.obs if hasattr(obs, "obs") else obs
        (logits, taus, old_logits), hidden = model(
            obs_next, sample_size=sample_size, state=state, info=batch.info
        )

        q = self.compute_q_value(logits, getattr(obs, "mask", None))
        old_q = self.compute_q_value(old_logits, getattr(obs, "mask", None))

        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[1]
        act = to_numpy(q.max(dim=1)[1])
        old_act = to_numpy(old_q.max(dim=1)[1])
        # print("N ACTIONS ARE ", q.shape[1])
        # print("OBS SHAPE IS ", obs.shape)
        # print("ACTION MASK IS ", getattr(obs[0], "mask", None))
        # print("NEW Q IS ", q[0, :])
        # print("OLD Q IS ", old_q[0, :])

        # print("NEW ACT IS ", act)
        # print("OLD ACT IS ", old_act)
        # assert (act==old_act).all(), "ACTIONS NOT EQUAL"
        return Batch(logits=logits, act=act, state=hidden, taus=taus)