import torch
from torchvision import datasets, transforms
from .Cifar10 import get_cifa10
from .TinyImageNet import get_tinyimagenet
from .SVHN import get_svhn
from .ImageNet import get_imagenet
# from .ImageNet_ddp import get_imagenet_ddp

def get_data_loader(name, train, split, batch_size, image_size):
    """Get data loader by name."""
    if name is "cifar10":
        return get_cifa10(train, split, batch_size, image_size)
    elif name == "tinyimagenet":
        return get_tinyimagenet(train, split, batch_size, image_size)
    elif name == "svhn":
        return get_svhn(train, split, batch_size, image_size) 
    elif name == "imagenet":
        return get_imagenet(train, split, batch_size, image_size) 
    # elif name == "imagenet_ddp":
    #     return get_imagenet_ddp(args, train, split, batch_size, image_size) 