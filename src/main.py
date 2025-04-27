import torch
from torch.utils.data import Dataset, DataLoader

class PaUDataset(Dataset):
    """
    Dataset wrapping parent and noise inputs along with targets for training an MLP.
    """
    def __init__(self, pa_data, u_data, y_data):
        """
        Args:
            pa_data (dict[str, torch.Tensor]): parent variable tensors of shape (N, dim).
            u_data (dict[str, torch.Tensor]): noise variable tensors of shape (N, dim).
            y_data (torch.Tensor): target tensor of shape (N, output_dim).
        """
        self.pa_data = pa_data
        self.u_data = u_data
        self.y_data = y_data
        self.length = y_data.size(0)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        pa = {k: v[idx] for k, v in self.pa_data.items()}
        u = {k: v[idx] for k, v in self.u_data.items()}
        y = self.y_data[idx]
        return pa, u, y

def train_mlp(
    model,
    pa_data,
    u_data,
    y_data,
    epochs=100,
    batch_size=32,
    lr=1e-3,
    criterion=None,
    device=None,
    verbose=True
):
    """
    Train a PyTorch MLP model on given parent and noise inputs.

    Args:
        model (torch.nn.Module): an instance of the MLP class.
        pa_data (dict[str, torch.Tensor]): parent input tensors, each of shape (N, dim).
        u_data (dict[str, torch.Tensor]): noise input tensors, each of shape (N, dim).
        y_data (torch.Tensor): target tensor of shape (N, output_dim).
        epochs (int): number of training epochs.
        batch_size (int): batch size for DataLoader.
        lr (float): learning rate for Adam optimizer.
        criterion (torch.nn.Module, optional): loss function. If None, defaults to
            BCELoss for single-output sigmoid MLP, else MSELoss.
        device (str or torch.device, optional): device to train on.
        verbose (bool): whether to print progress.

    Returns:
        model: the trained model.
    """
    # Device setup
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
    model.to(device)

    # Loss function
    if criterion is None:
        # default: binary cross-entropy for single-output sigmoid, else MSE
        out_dim = getattr(model, 'o_size', None)
        if out_dim == 1:
            criterion = torch.nn.BCELoss()
        else:
            criterion = torch.nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # DataLoader
    dataset = PaUDataset(pa_data, u_data, y_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for pa_batch, u_batch, y_batch in loader:
            # Move data to device
            pa_batch = {k: v.to(device) for k, v in pa_batch.items()}
            u_batch = {k: v.to(device) for k, v in u_batch.items()}
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(pa_batch, u_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y_batch.size(0)

        if verbose:
            avg_loss = total_loss / len(dataset)
            print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

    return model