import torch
from torch.utils.data import Dataset

class NCMDataset(Dataset):
    def __init__(self, df, variables):
        self.df = df.reset_index(drop=True)
        self.variables = variables

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = {}
        for v in self.variables:
            val = self.df.loc[idx, v]
            tensor = torch.tensor(val, dtype=torch.float)
            if tensor.ndim == 0:
                tensor = tensor.unsqueeze(0)
            sample[v] = tensor
        return sample