import torch as T
import torch.nn as nn

from src.scm.distribution import *

class SCM(nn.Module):
    def __init__(self, v, f, pu: Distribution):
        super().__init__()
        self.v = v
        self.u = list(pu)
        self.f = f
        self.pu = pu
        self.device_param = nn.Parameter(T.empty(0))
    
    # TODO