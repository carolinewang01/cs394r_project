from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.distributions.normal import Normal
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
        eta: float=1., 
        risk_distortion=None,
        device: Union[str, int, torch.device] = "cpu"
    ) -> None:
        super().__init__(preprocess_net=preprocess_net, 
            action_shape=action_shape, 
            hidden_sizes=hidden_sizes, 
            num_cosines=num_cosines, 
            preprocess_net_output_dim=preprocess_net_output_dim, 
            device=device
            )

        # assert 0 < eta <= 1, "eta must be in (0, 1]"
        assert risk_distortion in ["cvar", "wang", "pow", None]
        self.eta = eta
        self.risk_distortion = risk_distortion

    def transform_tau(self, taus): 
        # taus shape is (batch_size, 8)
        if self.risk_distortion is None:
            risk_measure =  taus
            # risk_measure =  torch.exp(taus)

        elif self.risk_distortion == "cvar":
            # print("TAU SHAPE IS ", taus.shape)
            risk_measure = self.eta * taus
        elif self.risk_distortion == "wang":
            normal = Normal(0, 1)
            risk_measure = normal.cdf(normal.icdf(taus) + self.eta)
        elif self.risk_distortion == "pow":
            if self.eta >= 0: # risk seeking
                risk_measure = torch.pow(taus, 1/(1+np.abs(self.eta)))
            else: # risk averse
                ones = torch.ones_like(taus)
                risk_measure = ones - torch.pow(ones - taus, 1/(1+np.abs(self.eta)))
        return risk_measure

    def forward(self, 
        obs: Union[np.ndarray, torch.Tensor], 
        sample_size: int, 
        **kwargs: Any
    ) -> Tuple[Any, torch.Tensor]:
        r"""Mapping: s -> Q(s, \*)."""
        logits, hidden = self.preprocess(obs, state=kwargs.get("state", None))
        # Sample fractions.
        batch_size = logits.size(0)
        taus = torch.rand(
            batch_size, sample_size, dtype=logits.dtype, device=logits.device
        )
        new_taus = self.transform_tau(taus)
        # print("EMBEDDED NEW TAU ", torch.mean(self.embed_model(new_taus)))
        embedding = (logits.unsqueeze(1) *
                     self.embed_model(new_taus)).view(batch_size * sample_size, -1)

        out = self.last(embedding).view(batch_size, sample_size, -1).transpose(1, 2)
        ### 
        # print("EMBEDDED OLD TAU ", torch.mean(self.embed_model(taus)))
        old_embedding = (logits.unsqueeze(1) *
                     self.embed_model(taus)).view(batch_size * sample_size, -1)
        old_out = self.last(old_embedding).view(batch_size, sample_size, -1).transpose(1, 2)

        return (out, new_taus, old_out), hidden
