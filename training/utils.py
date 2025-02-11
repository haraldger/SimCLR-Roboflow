import os

import torch
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.io import read_image

def get_optimizer(name, model, lr, weight_decay):
    if name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f'Optimizer {name} not supported')
    
def get_backbone(size):
    if size == 18:
        return resnet18()
    elif size == 34:
        return resnet34()
    elif size == 50:
        return resnet50()
    elif size == 101:
        return resnet101()
    elif size == 152:
        return resnet152()
    else:
        raise ValueError(f'Model size {size} not supported')
    
def get_normalization(data_path):
    images = torch.tensor([])

    for image_file in os.listdir(data_path):
        image = read_image(os.path.join(data_path, image_file))
        images = torch.cat((images, image.unsqueeze(0)), dim=0)
    
    mean = images.mean(dim=(0, 2, 3))
    std = images.std(dim=(0, 2, 3))

    return {"mean": mean, "std": std}