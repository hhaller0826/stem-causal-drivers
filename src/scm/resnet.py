import torch
import torch.nn as nn

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
