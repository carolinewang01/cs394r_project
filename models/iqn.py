from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from torch import nn
from tianshou.utils.net.discrete import ImplicitQuantileNetwork


class RiskAwareIQN(ImplicitQuantileNetwork):
    '''
    modifies the base IQN network to allow for risk sensitive policies
    based on https://github.com/thu-ml/tianshou/blob/v0.4.7/tianshou/utils/net/discrete.py#L158

    tau_upper_lim: a float in (0, 1]. 
    '''
    def __init__(self,
        preprocess_net: nn.Module,
        action_shape: Sequence[int],
        hidden_sizes: Sequence[int] = (),
        num_cosines: int = 64,
        preprocess_net_output_dim: Optional[int] = None,
        cvar_eta: float=1., 
        device: Union[str, int, torch.device] = "cpu"
    ) -> None:
        super().__init__(preprocess_net=preprocess_net, 
            action_shape=action_shape, 
            hidden_sizes=hidden_sizes, 
            num_cosines=num_cosines, 
            preprocess_net_output_dim=preprocess_net_output_dim, 
            device=device
            )

        assert 0 < cvar_eta <= 1, "cvar_eta must be in (0, 1]"
        self.cvar_eta = cvar_eta

    def forward(self, 
        obs: Union[np.ndarray, torch.Tensor], 
        sample_size: int, 
        risk_aware: bool=False,
        **kwargs: Any
    ) -> Tuple[Any, torch.Tensor]:
        r"""Mapping: s -> Q(s, \*)."""
        logits, hidden = self.preprocess(obs, state=kwargs.get("state", None))
        # Sample fractions.
        batch_size = logits.size(0)
        taus = torch.rand(
            batch_size, sample_size, dtype=logits.dtype, device=logits.device
        )
        if risk_aware:
            taus *= cvar_eta
        embedding = (logits.unsqueeze(1) *
                     self.embed_model(taus)).view(batch_size * sample_size, -1)
        out = self.last(embedding).view(batch_size, sample_size, -1).transpose(1, 2)
        return (out, taus), hidden
