import torch as T
import torch.nn as nn

from src.scm.scm import SCM
from src.scm.distribution import *

"""NOTE: WE MAY NEED TO CHANGE THIS
When initializing FF_NCM and MLP they have things like u_size, v_size, o_size 
These correspond to the "size" of a single node. I think that this means like if node Z={"age","sex","race"} then its size would be 3. 
Right now I am keeping everything as size=1 and we might just have to change that later.
"""

class NN_for_in_between_nodes(nn.Module):
    # TODO: they have an MLP class
    def __init__(self, pa_size, u_size, h_size=128, *args, **kwargs):
        super().__init__(*args, **kwargs)


class FF_NCM(SCM):
    def __init__(self, cg, f={}, hyperparams=None):
        self.cg = cg
        super().__init__(
            v = list(cg),
            f=nn.ModuleDict({
                v: f[v] if v in f else NN_for_in_between_nodes(
                    pa_size=len(cg.pa[v]),
                    u_size=len(cg.v2c2[v])
                )
                for v in cg
            }),
            pu=UniformDistribution(cg.c2, len(cg.c2))
        )
        # TODO

    