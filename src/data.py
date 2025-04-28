# import torch
# from torch.utils.data import Dataset

# class NCMDataset(Dataset):
#     def __init__(self, df, variables):
#         self.df = df.reset_index(drop=True)
#         self.variables = variables

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         sample = {}
#         for v in self.variables:
#             val = self.df.loc[idx, v]
#             tensor = torch.tensor(val, dtype=torch.float)
#             if tensor.ndim == 0:
#                 tensor = tensor.unsqueeze(0)
#             sample[v] = tensor
#         return sample

import torch
from torch.utils.data import Dataset

class NCMDataset(Dataset):
    def __init__(self, X, y):
        """
        X: pandas DataFrame or numpy array (n_samples x n_features)
        y: pandas Series/1D array (n_samples,) or numpy array
        """
        # If pandas, reset index to ensure continuous integer indexing
        try:
            self.X = X.reset_index(drop=True)
            self.y = y.reset_index(drop=True)
        except AttributeError:
            # assume they’re numpy arrays
            self.X = X
            self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # get features
        x = self.X.loc[idx].values if hasattr(self.X, 'loc') else self.X[idx]
        x = torch.tensor(x, dtype=torch.float)

        # get target
        y = self.y.loc[idx] if hasattr(self.y, 'loc') else self.y[idx]
        y = torch.tensor(y, dtype=torch.float)

        # if y is a scalar, make it a 1-element vector
        if y.ndim == 0:
            y = y.unsqueeze(0)

        return x, y
