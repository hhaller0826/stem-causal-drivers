import torch as T
import torch.nn as nn

from src.scm.scm import SCM
from src.scm.distribution import *
from src.scm.mlp import *

"""NOTE: WE MAY NEED TO CHANGE THIS
When initializing FF_NCM and MLP they have things like u_size, v_size, o_size 
These correspond to the "size" of a single node. I think that this means like if node Z={"age","sex","race"} then its size would be 3. 
Right now I am keeping everything as size=1 and we might just have to change that later.
"""

class FF_NCM(SCM):
    def __init__(self, cg, v_size={}, default_v_size=1, u_size={},
                 default_u_size=1, f={}, hyperparams=None):
        if hyperparams is None:
            hyperparams = dict()

        self.cg = cg
        self.u_size = {k: u_size.get(k, default_u_size) for k in self.cg.c2}
        self.v_size = {k: v_size.get(k, default_v_size) for k in self.cg}
        super().__init__(
            v=list(cg),
            f=nn.ModuleDict({
                v: f[v] if v in f else MLP(
                    {k: self.v_size[k] for k in self.cg.pa[v]},
                    {k: self.u_size[k] for k in self.cg.v2c2[v]},
                    self.v_size[v],
                    h_size=hyperparams.get('h-size', 128)
                )
                for v in cg}),
            pu=UniformDistribution(self.cg.c2, self.u_size))

    def convert_evaluation(self, samples):
        return {k: T.round(samples[k]) for k in samples}

    