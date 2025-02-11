import torch

def training_epoch(data_objects, training_objects):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epoch_loss = 0

    # Unpack objects
    dataloader = data_objects["training_dataloader"]
    model = training_objects["model"]
    loss_fn = training_objects["loss_fn"]
    optimizer = training_objects["optimizer"]
    scheduler = training_objects["scheduler"]

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

def validation_epoch(data_objects, training_objects):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epoch_loss = 0

    # Unpack objects
    dataloader = data_objects["validation_dataloader"]
    model = training_objects["model"]
    loss_fn = training_objects["loss_fn"]

    model.eval()
    with torch.no_grad():
        for idx, ((images0, images1), _, _) in enumerate(dataloader):
            images0 = images0.to(device)
            images1 = images1.to(device)
            
            # Forward pass
            embeddings0 = model(images0)
            embeddings1 = model(images1)

            # Compute loss
            loss = loss_fn(embeddings0, embeddings1)
            epoch_loss += loss.item()
    
    epoch_loss /= len(dataloader)
    return epoch_loss