import argparse

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from lightly.transforms import SimCLRTransform
from lightly.data import LightlyDataset
from lightly.loss import NTXentLoss

from models.CLRNet import CLRNet
from training.epochs import training_epoch, validation_epoch
from training.utils import get_optimizer, get_normalization

def main(config):
    training_data_path = 'data/train/images'
    validation_data_path = 'data/valid/images'

    # Configure data objects
    normalization = get_normalization(training_data_path)
    transforms = SimCLRTransform(input_size=640, normalize=normalization)

    training_dataset = LightlyDataset(input_dir=training_data_path, transform=transforms)
    training_dataloader = DataLoader(training_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True, num_workers=0)

    validation_dataset = LightlyDataset(input_dir=validation_data_path, transform=transforms)
    validation_dataloader = DataLoader(validation_dataset, batch_size=config["batch_size"], shuffle=False, drop_last=False, num_workers=0)

    # Configure training objects
    model = CLRNet(size=config["model_size"])
    loss_fn = NTXentLoss(temperature=config["tau"])
    optimizer = get_optimizer(config["optimizer"], model, config["learning_rate"], config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=len(training_dataloader))

    # Train the model
    best_loss = float('inf')
    for epoch in range(config["epochs"]):
        print(f'Epoch {epoch}/{config["epochs"]}')

        # Train the model
        training_loss = training_epoch(model, training_dataloader, loss_fn, optimizer, scheduler)
        print(f'Training Loss: {training_loss}')

        # Evaluate the model
        validation_loss = validation_epoch(model, validation_dataloader, loss_fn)
        print(f'Validation Loss: {validation_loss}')

        # Save the model if validation loss is the best we've seen so far
        if validation_loss < best_loss:
            print('Validation loss decreased. Saving model...')
            best_loss = validation_loss
            torch.save(model.state_dict(), config["output_path"])

    print(f'Training complete. Model saved to {config["output_path"]}')



if __name__ == "__main__":
    config = dict()
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model_size', type=int, default=18, choices=[18, 34, 50, 101, 152])
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--learning_rate', type=float, default=0.01875)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--output_path', type=str, default='model.pt')
    args = parser.parse_args()

    config['epochs'] = args.epochs
    config['model_size'] = args.model_size
    config['batch_size'] = args.batch_size
    config['optimizer'] = args.optimizer
    config['learning_rate'] = args.learning_rate
    config['weight_decay'] = args.weight_decay
    config['tau'] = args.tau
    config['output_path'] = args.output_path
    
    main(config)