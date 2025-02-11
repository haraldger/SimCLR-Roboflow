from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from lightly.transforms import SimCLRTransform
from lightly.data import LightlyDataset
from lightly.loss import NTXentLoss

from models.CLRNet import CLRNet
from training.utils import get_optimizer, get_normalization

def data_factory(config):
    training_data_path = 'data/train/images'
    validation_data_path = 'data/valid/images'

    # Configure data objects
    normalization = get_normalization(training_data_path)
    transforms = SimCLRTransform(input_size=640, normalize=normalization)

    training_dataset = LightlyDataset(input_dir=training_data_path, transform=transforms)
    training_dataloader = DataLoader(training_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True, num_workers=0)

    validation_dataset = LightlyDataset(input_dir=validation_data_path, transform=transforms)
    validation_dataloader = DataLoader(validation_dataset, batch_size=config["batch_size"], shuffle=False, drop_last=False, num_workers=0)

    return {
        "training_dataloader": training_dataloader,
        "validation_dataloader": validation_dataloader
    }

def training_factory(config, objects):
    training_dataloader = objects["training_dataloader"]

    # Configure training objects
    model = CLRNet(size=config["model_size"])
    loss_fn = NTXentLoss(temperature=config["tau"])
    optimizer = get_optimizer(config["optimizer"], model, config["learning_rate"], config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=len(training_dataloader))

    return {
        "model": model,
        "loss_fn": loss_fn,
        "optimizer": optimizer,
        "scheduler": scheduler
    }