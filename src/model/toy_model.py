import torch
import torch.nn as nn
from typing import List, Tuple, Union, Optional


# Toy Architectures
class MinValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = nn.Parameter(torch.zeros(1))

    def forward(self):
        return self.m


class MLP(nn.Module):
    def __init__(self, *hidden_dims: int):
        """Sequential linear layers with the ReLU activation.
        
        ReLU is applied between all layers. A number of layers equals
        `len(hidden_dims) - 1`. The first and the last hidden dims are treated as the 
        input and the output dimensions of the backbone.
        """
        assert len(hidden_dims) >= 2
        super().__init__()
        
        inp, *hidden_dims = hidden_dims
        self._layers = nn.Sequential(nn.Linear(inp, hidden_dims[0]))
        for inp, out in zip(hidden_dims[:-1], hidden_dims[1:]):
            self._layers.append(nn.ReLU(inplace=True))
            self._layers.append(nn.Linear(inp, out))
    
    def forward(self, x): return self._layers(x)


class OTMap(nn.Module):
    def __init__(
        self,
        inp_dim: int = None,
        hidden_dims: List[int] = None,
        out_dim: int = None,
        *args, **kwargs,
    ):
        """Initialize OT map class.
        
        Args:
            inp_dim: a dimensionality of the source space.
            out_dim: a dimensionality of the target space.
            hidden_dims: hidden dimensions.
        """
        super().__init__()
        
    def forward(
        self, 
        x: torch.FloatTensor,
        reg: bool = False,
    ) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]]:
        """Compute OT Map.
        
        If the map is weak, return one sample per input item.
        
        Args:
            x: tensor of shape (bs, inp_dim)
            reg: wether to return the regularization term
        
        Returns:
            tensor of shape (bs, out_dim) [and regularization term]
        """
        
        raise NotImplementedError


class DeterministicMap(OTMap):
    def __init__(self, inp_dim: int, hidden_dims: List[int], out_dim: int):
        super().__init__()
        self._bb = MLP(inp_dim, *hidden_dims, out_dim)
        
    def forward(self, x, reg: bool = False):
        out = self._bb(x)
        if reg:
            return out, torch.tensor(0.0, device=x.device)
        return out


class NoiseInputMap(OTMap):
    def __init__(
        self,
        inp_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        prior: torch.distributions.Distribution,
        noise_dim: Optional[int] = None,
    ):
        super().__init__()
        self._noise_dim = noise_dim or inp_dim
        self._prior = prior
        self._bb = MLP(inp_dim + self._noise_dim, *hidden_dims, out_dim)
        
    def forward(self, x, reg: bool = False):
        bs = x.shape[0]
        dev = x.device
        
        noise = torch.randn(bs, self._noise_dim, device=dev)
        x = torch.cat((x, noise), dim=-1)
        out = self._bb(x)
        ed = self.energy_dist_reg_sample(out)
        
        if reg:
            return out, ed
        return out
        
    def energy_dist_reg_sample(
        self,
        sample: torch.FloatTensor,
    ):
        """Compute energy distance (only sample-dependent terms) using sample estimate.

        Args:
            sample: has shape (bs, d)
            prior: torch distribution of item shape (d,)

        Returns:
            tensor of shape (bs,)
        """
        pr_sample_1, pr_sample_2 = self._prior.sample((2, *sample.shape[:-1]))
        l12 = (sample - pr_sample_1).norm(dim=1)
        l11 = (pr_sample_1 - pr_sample_2).norm(dim=1)
        return 2 * l12 - l11


def get_pots(CONFIG):
    assert len(CONFIG.LAMBDAS) == CONFIG.K

    if CONFIG.K == 2:
        pots = Two_Pots
    elif CONFIG.K > 2:
        pots = Multi_Pots

    return pots(CONFIG.LAMBDAS, 
                CONFIG.INPUT_DIM,
                *CONFIG.HIDDEN_DIMS,
                CONFIG.OUTPUT_DIM_POT
                ).to(CONFIG.DEVICE)


class Pots(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def __getitem__(self, idx):
        pass


class Two_Pots(Pots):
    # TODO: optimize when 2 potentials
    def __init__(self, bary_weights, *dims):
        super().__init__()
        assert len(bary_weights) > 1
        self._lambdas = bary_weights
        self._net = MLP(*dims)
        
    def __getitem__(self, idx):
        assert 0 <= idx < 2 # only when there are two prob
        
        if idx == 0:
            def f_pot(x, m=None): # include m
                res = 0.0
                res += self._net(x)
                if m is not None:
                    res += m / len(self._lambdas) / self._lambdas[idx] # include m
                return res
        else:
            def f_pot(x, m=None): # include m
                res = 0.0
                res -= self._net(x)
                if m is not None:
                    res += m / len(self._lambdas) / self._lambdas[idx] # include m
                return res

        return f_pot


class Multi_Pots(Pots):
    def __init__(self, bary_weights, *dims):
        super().__init__()
        assert len(bary_weights) > 1
        self._lambdas = bary_weights
        self._nets = nn.ModuleList([MLP(*dims) for _ in range(len(bary_weights))])
        
    def __getitem__(self, idx):
        assert 0 <= idx < len(self._lambdas)
        
        def f_pot(x, m=None): # include m
            res = 0.0
            for i, (net, lmbd) in enumerate(zip(self._nets, self._lambdas)):

                if i == idx:
                    res += net(x)
                    if m is not None:
                        res += m / len(self._lambdas) / self._lambdas[idx] # include m
                else:
                    res -= lmbd * net(x) / (len(self._lambdas) - 1) / self._lambdas[idx]
            return res
        
        return f_pot


