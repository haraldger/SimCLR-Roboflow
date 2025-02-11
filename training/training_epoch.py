import torch

def training_epoch(model, dataloader, loss_fn, optimizer, scheduler=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epoch_loss = 0

    model.train()
    for idx, ((images0, images1), _, _) in enumerate(dataloader):
        images0 = images0.to(device)
        images1 = images1.to(device)

        # Forward pass
        embeddings0 = model(images0)
        embeddings1 = model(images1)

        # Compute loss
        loss = loss_fn(embeddings0, embeddings1)
        epoch_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if scheduler is not None:
        scheduler.step()

    epoch_loss /= len(dataloader)
    return epoch_loss

