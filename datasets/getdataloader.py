from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from torchvision.transforms import *
from torchvision.datasets import CIFAR10, CIFAR100



import torch
import os
from datasets.augment import Cutout, CIFAR10Policy

# your own data dir
DIR = {'CIFAR10': '../data', 'CIFAR100': '../data', 'ImageNet': '../data/', 'COCO': '../data/', }

def GetCifar10(args):
    trans_t = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        Cutout(n_holes=1, length=16)
    ])
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_data = CIFAR10(args.dataset_path, train=True, transform=trans_t, download=True)
    test_data = CIFAR10(args.dataset_path, train=False, transform=trans, download=True)
    l_tr = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=8)
    l_te = DataLoader(test_data, batch_size=args.batchsize, shuffle=False, num_workers=8)
    return l_tr, l_te

def GetCifar100(args):
    trans_t = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        CIFAR10Policy(),
        ToTensor(),
        Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]],
                  std=[n/255. for n in [68.2,  65.4,  70.4]]),
        Cutout(n_holes=1, length=16)
    ])
    trans = Compose([transforms.ToTensor(), transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]])])
    train_data = CIFAR100(args.dataset_path, train=True, transform=trans_t, download=True)
    test_data = CIFAR100(args.dataset_path, train=False, transform=trans, download=True)
    l_tr = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=8, pin_memory=True)
    l_te = DataLoader(test_data, batch_size=args.batchsize, shuffle=False, num_workers=4, pin_memory=True)
    return l_tr, l_te


def create_dataloader(dataset, batch_size, shuffle, num_workers, distributed):
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=sampler,
            pin_memory=torch.cuda.is_available()
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

def GetImageNet(args):
    if 'resnet' in args.model_name or 'vgg' in args.model_name:
        trans = transforms.Compose([
                    transforms.Resize(size=235, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                    transforms.CenterCrop(size=(224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
    elif 'vit' in args.model_name:
        trans = transforms.Compose([
                    transforms.Resize(size=248, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                    transforms.CenterCrop(size=(224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
    elif args.model_name in ['eva02_tiny','eva02_small']: ##'eva'
        trans = transforms.Compose([
                    transforms.Resize(size=336, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                    transforms.CenterCrop(size=(336, 336)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4815, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758]),
                ])
    elif args.model_name in ['eva02_base','eva02_large']: ##'eva'
        trans = transforms.Compose([
                    transforms.Resize(size=448, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                    transforms.CenterCrop(size=(448, 448)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4815, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758]),
                ])

    if args.mode in ['train_snn']:
        train_data = datasets.ImageFolder(root=os.path.join(args.dataset_path, 'train'), transform=trans)
        l_tr = create_dataloader(train_data, args.batchsize, shuffle=True, num_workers=8, distributed=args.distributed)
    else:
        l_tr = None

    test_data = datasets.ImageFolder(root=os.path.join(args.dataset_path, 'val'), transform=trans)
    l_te = create_dataloader(test_data, args.batchsize, shuffle=False, num_workers=2, distributed=args.distributed)


    return l_tr, l_te

from torchvision.datasets import CocoDetection
class ComposeTransforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image = t(image)
        return image, target

def GetCOCO(args):
    transform = ComposeTransforms([
        transforms.ToTensor(),
    ])
    train_loader = None
    val_dataset = CocoDetection(
        root=os.path.join(args.dataset_path, 'COCO/val2017'),
        annFile=os.path.join(args.dataset_path, 'COCO/annotations/instances_val2017.json'),
        transforms=transform
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )
    return train_loader, val_loader
