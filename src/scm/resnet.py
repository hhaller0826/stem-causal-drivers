import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbedResNetMLP(nn.Module):
    def __init__(self, pa_spec, u_dim, out_dim, hidden_dim=64, emb_dim=8):
        """
        pa_spec: dict {var_name: ('num',) or ('cat', n_categories)}
        u_dim: dimension of noise vector for this node
        out_dim: 1 for scalar/continuous or binary
        """
        super().__init__()
        # 1) build embeddings + numeric scalers
        self.embeddings = nn.ModuleDict({
            v: nn.Embedding(n_cat, emb_dim)
            for v, spec in pa_spec.items() if spec[0]=='cat'
        })
        self.num_vars = [v for v,s in pa_spec.items() if s[0]=='num']
        # total input dim D = emb_dim*num_cat_vars + len(num_vars) + u_dim
        D = emb_dim*sum(1 for s in pa_spec.values() if s[0]=='cat') \
            + len(self.num_vars) + u_dim

        # 2) residual projection
        self.proj = nn.Linear(D, hidden_dim)

        # 3) two-layer core
        self.l1 = nn.Linear(D, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(0.1)

        # 4) output head
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, pa, u):
        # pa: dict of parent tensors, pa[v] shape (B,1) for both num & cat (indexes)
        emb_parts, num_parts = [], []
        for v in self.embeddings:
            emb_parts.append(self.embeddings[v](pa[v].long()))
        for v in self.num_vars:
            num_parts.append(pa[v].float())
        x = torch.cat(emb_parts + num_parts + [u], dim=1)  # (B, D)

        x_proj = self.proj(x)                             # (B, H)
        h = F.relu(self.ln1(self.l1(x)))
        h = F.relu(self.ln2(self.l2(h)))
        r = self.drop(h + x_proj)                         # residual + dropout

        out = self.out(r)
        return out if out.shape[1]>1 else out.squeeze(1)


class ResNet(nn.Module):
    def __init__(self, pa_size: dict, u_size: dict, o_size: int, h_size: int = 64, dropout: float = 0.1, pa_embedding_specs: dict = None):
        super().__init__()

        self.pa = sorted(pa_size)
        self.set_pa = set(self.pa)
        self.u = sorted(u_size)
        self.pa_size = pa_size
        self.u_size = u_size
        self.o_size = o_size
        self.h_size = h_size

        # pa_embedding_specs[k] = (num_categories, embedding_dim)
        self.pa_embedding_specs = pa_embedding_specs or {}
        self.embeddings = nn.ModuleDict({
            k: nn.Embedding(num_cat, emb_dim)
            for k, (num_cat, emb_dim) in self.pa_embedding_specs.items()
        })

        # pa vars that remain “numeric” (no embedding)
        self.num_pa = [k for k in self.pa if k not in self.pa_embedding_specs]

        # compute total input size = numeric dims + embedding dims + sum(u dims)
        total_numeric = sum(pa_size[k] for k in self.num_pa)
        total_emb = sum(emb_dim for (_, emb_dim) in self.pa_embedding_specs.values())
        total_u = sum(u_size[k]  for k in self.u)
        self.i_size = total_numeric + total_emb + total_u

        # residual projection
        self.proj = nn.Linear(self.i_size, self.h_size)

        # two-layer core
        self.l1 = nn.Linear(self.i_size, self.h_size)
        self.ln1 = nn.LayerNorm(self.h_size)
        self.l2 = nn.Linear(self.h_size, self.h_size)
        self.ln2 = nn.LayerNorm(self.h_size)
        self.drop = nn.Dropout(dropout)

        # final output head
        self.out = nn.Linear(self.h_size, self.o_size)

        self.device_param = nn.Parameter(torch.empty(0))

        # init weights using xavier
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, pa: dict, u: dict, include_inp: bool = False):
        parts = []

        for k in self.pa_embedding_specs:
            idx = pa[k].long().squeeze(-1)
            parts.append(self.embeddings[k](idx))

        for k in self.num_pa:
            parts.append(pa[k].float())

        if len(self.u) > 0:
            parts.append(torch.cat([u[k] for k in self.u], dim=1))

        inp = torch.cat(parts, dim=1)

        # residual projection
        x_proj = self.proj(inp)

        # core
        h = F.relu(self.ln1(self.l1(inp)))
        h = F.relu(self.ln2(self.l2(h)))
        r = self.drop(h + x_proj)

        # output
        out = self.out(r)
        out = torch.sigmoid(out)

        if include_inp:
            return out, inp
        return out