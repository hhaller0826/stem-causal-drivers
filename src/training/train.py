import torch as T

from torch.utils.data._utils.collate import default_collate
from src.training.divergence import MMD_loss

def train_ncm(model, dataloader, optimizer, device, num_epochs=10):
    model.to(device)
    model.train()
    ordered_v = model.v

    for epoch in range(1, num_epochs+1):
        epoch_loss = 0.0
        for batch in dataloader:
            # if DataLoader gives you back a list of samples, collate it
            if isinstance(batch, list):
                batch = default_collate(batch)

            # now batch is a dict of batched tensors
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_size = next(iter(batch.values())).shape[0]

            ncm_batch = model(batch_size)
            data_matrix = T.cat([batch[k] for k in ordered_v], axis=1)
            ncm_matrix = T.cat([ncm_batch[k] for k in ordered_v], axis=1)

            optimizer.zero_grad()
            loss = MMD_loss(data_matrix.float(),ncm_matrix,gamma=1)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}")

    return model

def compute_accuracy(model, dataloader, device, target_var):
    model.eval()
    with T.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_size = next(iter(batch.values())).shape[0]

            preds=model(n=batch_size,select={target_var})[target_var]
            labels = batch[target_var]
            
            labels = T.cat([batch[target_var]], axis=1)
            pred_labels = T.cat([preds], axis=1)
            loss = MMD_loss(labels,pred_labels)
    return 1-loss

def print_accuracy(var, trained_ncm, train_dataloader, test_dataloader):
    train_acc = compute_accuracy(trained_ncm, train_dataloader, 'cpu', var)
    print(f'Final train accuracy for {var}: {train_acc:.4f}')

    test_acc = compute_accuracy(trained_ncm, test_dataloader, 'cpu', var)
    print(f'Final test accuracy  for {var}: {test_acc:.4f}')

