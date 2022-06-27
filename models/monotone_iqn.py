from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from tianshou.utils.net.discrete import ImplicitQuantileNetwork

# TODO: define hypernetwork
# input to hypernetwork should be logits
# output of hypernetwork should be embed network weights
# input of embed network should be tau (potentially embedded with cosines?)
# TODO: replace embedding net with hypernetwork

class MonotoneIQN(nn.Module):
    """
    modifies the base IQN network to make it monotone in tau
    based on https://github.com/thu-ml/tianshou/blob/v0.4.7/tianshou/utils/net/discrete.py#L158

    """
    def __init__(
        self,
        preprocess_net: nn.Module,
        action_shape: Sequence[int],
        # hidden_sizes: Sequence[int] = (),
        # num_cosines: int = 64,
        sample_size: int, 
        preprocess_net_output_dim: Optional[int] = None,
        device: Union[str, int, torch.device] = "cpu"
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net

        # define in/out dims
        self.state_dim = getattr(
            preprocess_net, "output_dim", preprocess_net_output_dim
        )
        self.hypernet_embed_dim = 32 # 64
        self.sample_size = sample_size
        self.output_dim = np.prod(action_shape)

        # hypernet 
        self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, self.hypernet_embed_dim),
                                       nn.ReLU(),
                                       nn.Linear(self.hypernet_embed_dim, self.hypernet_embed_dim * self.sample_size))
        self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, self.hypernet_embed_dim),
                                       nn.ReLU(),
                                       nn.Linear(self.hypernet_embed_dim, self.hypernet_embed_dim * self.output_dim * self.sample_size))

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.hypernet_embed_dim)

        # V(s) instead of a bias for the last layers
        self.hyper_b_final = nn.Linear(self.state_dim, self.output_dim * self.sample_size)
                               # nn.ReLU(),
                               # nn.Linear(self.hypernet_embed_dim, 1))

    def forward(  # type: ignore
        self, obs: Union[np.ndarray, torch.Tensor], sample_size: int, **kwargs: Any
    ) -> Tuple[Any, torch.Tensor]:
        r"""Mapping: s -> Q(s, \*).
        A difference from the baseline IQN implementation: sample size argument is ignored 
        because we must fix the sample size in order to specify the hypernetwork structure. 
        """
        # obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        logits, hidden = self.preprocess(obs, state=kwargs.get("state", None))
        # Sample fractions.
        batch_size = logits.size(0)
        taus = torch.rand(
            batch_size, self.sample_size, dtype=logits.dtype, device=logits.device
        ) # TODO: decide whether or not to embed taus?

        # First layer
        # print("STATE DIM IS ", self.state_dim)
        # print("OBS SHAPE IS ", obs.shape)
        w1 = torch.abs(self.hyper_w_1(logits))
        b1 = self.hyper_b_1(logits)
        w1 = w1.view(-1, self.sample_size, self.hypernet_embed_dim)
        b1 = b1.view(-1, 1, self.hypernet_embed_dim)
        taus = taus.view(-1, 1, self.sample_size)

        # print("BATCH SIZE IS ", batch_size)
        # print("SELF SAMPLE SIZE IS ", self.sample_size)

        # print("B1 SHAPE ", b1.shape)
        # print("W1 SHAPE ", w1.shape)
        # print("TAUS SHAPE ", taus.shape)
        x = torch.bmm(taus, w1)
        # print("X SHAPE  ", x.shape)
        x = x + b1
        # hidden = F.elu(x)
        hidden = F.relu(x)
        # print("HIDDEN SHAPE IS ", hidden.shape)
        # hidden = F.elu(torch.bmm(taus, w1) + b1) # TODO: CHECK ELU

        # Second layer
        w_final = torch.abs(self.hyper_w_final(logits))
        w_final = w_final.view(-1, self.hypernet_embed_dim, self.output_dim * self.sample_size)
        # print("W_FINAL SHAPE IS ", w_final.shape)
        # State-dependent bias
        b_final = self.hyper_b_final(logits).view(-1, 1, self.output_dim * self.sample_size)
        # Compute final output
        # print("B FINAL SHAPE IS ",b_final.shape)
        y = torch.bmm(hidden, w_final) + b_final
        # print("Y SHAPE IS ", y.shape)

        out = y.view(batch_size, self.output_dim, self.sample_size)
        return (out, taus), hidden