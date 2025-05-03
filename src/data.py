import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd

from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split


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
    
    def get_dataloader(self,batch_size=32,shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)


def tt_split(df, binary_cols, test_size=0.4, random_state=42):
    dat_train, dat_test = train_test_split(df, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    dat_train = scaler.fit_transform(dat_train)
    dat_test = scaler.fit_transform(dat_test)

    train_df = pd.DataFrame(dat_train, columns=df.columns)
    test_df = pd.DataFrame(dat_test, columns=df.columns)
    
    for col in binary_cols:
        train_df[col] = train_df[col].apply(lambda x: 0 if x <= 0 else 1)
        test_df[col] = test_df[col].apply(lambda x: 0 if x <= 0 else 1)
    
    return train_df, test_df