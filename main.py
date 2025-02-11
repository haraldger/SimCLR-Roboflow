import argparse

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from lightly.transforms import SimCLRTransform, utils
from lightly.data import LightlyDataset
from lightly.loss import NTXentLoss

from models.CLRNet import CLRNet
from training.training_epoch import training_epoch
from training.utils import get_optimizer, get_normalization

def main(config):
    training_data_path = 'data/train/images'

    # Configure data objects
    normalization = get_normalization(training_data_path)
    training_transforms = SimCLRTransform(input_size=640, normalize=normalization)
    training_dataset = LightlyDataset(input_dir=training_data_path, transform=training_transforms)
    training_dataloader = DataLoader(training_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True, num_workers=0)

    # Configure training objects
    model = CLRNet(size=config["model_size"])
    loss_fn = NTXentLoss(temperature=config["tau"])
    optimizer = get_optimizer(config["optimizer"], model, config["learning_rate"], config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=len(training_dataloader))

    # Train the model
    for epoch in range(config["epochs"]):
        epoch_loss = training_epoch(model, training_dataloader, loss_fn, optimizer, scheduler)
        print(f'Epoch {epoch}/{config["epochs"]}: Loss: {epoch_loss}')


if __name__ == "__main__":
    config = dict()
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model_size', type=int, default=18, choices=[18, 34, 50, 101, 152])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--learning_rate', type=float, default=0.01875)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--tau', type=float, default=0.1)
    args = parser.parse_args()

    config['epochs'] = args.epochs
    config['model_size'] = args.model_size
    config['batch_size'] = args.batch_size
    config['optimizer'] = args.optimizer
    config['learning_rate'] = args.learning_rate
    config['weight_decay'] = args.weight_decay
    config['tau'] = args.tau
    
    main(config)