import random

import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms

from timm.data import create_transform

from continual_datasets.continual_datasets import (
    MNIST_RGB,
    FashionMNIST,
    NotMNIST,
    SVHN,
    Flowers102,
    StanfordCars,
    CUB200,
    TinyImagenet,
    Scene67,
    Imagenet_R,
    Split_CZSL_Dataset,
)

import utils
from pathlib import Path


class Lambda(transforms.Lambda):
    def __init__(self, lambd, nb_classes):
        super().__init__(lambd)
        self.nb_classes = nb_classes

    def __call__(self, img):
        return self.lambd(img, self.nb_classes)


def target_transform(x, nb_classes):
    return x + nb_classes


def build_continual_dataloader(args):
    dataloader = list()
    class_mask = list() if args.task_inc or args.train_mask else None

    transform_train = build_transform(True, args)
    transform_val = build_transform(False, args)

    if args.dataset.startswith("czsl-"):
        splited_dataset, class_mask = get_dataset(
            args.dataset.replace("czsl-", ""),
            transform_train,
            transform_val,
            args,
        )
    elif args.dataset.startswith("Split-"):
        dataset_train, dataset_val = get_dataset(
            args.dataset.replace("Split-", ""),
            transform_train,
            transform_val,
            args,
        )
        args.nb_classes = len(dataset_val.classes)
        splited_dataset, class_mask = split_single_dataset(dataset_train, dataset_val, args)
    else:
        if args.dataset == "5-datasets":
            dataset_list = ["SVHN", "MNIST", "CIFAR10", "NotMNIST", "FashionMNIST"]
        else:
            dataset_list = args.dataset.split(",")

        if args.shuffle:
            random.shuffle(dataset_list)

        args.nb_classes = 0

    for i in range(args.num_tasks):
        if args.dataset.startswith("czsl-"):
            dataset_train, dataset_val = splited_dataset[i]
        elif args.dataset.startswith("Split-"):
            dataset_train, dataset_val = splited_dataset[i]
        else:
            dataset_train, dataset_val = get_dataset(dataset_list[i], transform_train, transform_val, args)
            transform_target = Lambda(target_transform, args.nb_classes)
            if class_mask is not None:
                class_mask.append([i + args.nb_classes for i in range(len(dataset_val.classes))])
                args.nb_classes += len(dataset_val.classes)
            if not args.task_inc:
                dataset_train.target_transform = transform_target
                dataset_val.target_transform = transform_target

        if args.distributed and utils.get_world_size() > 1:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()

            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train,
                num_replicas=num_tasks,
                rank=global_rank,
                shuffle=True,
            )

            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        dataloader.append({"train": data_loader_train, "val": data_loader_val})

    return dataloader, class_mask


def get_dataset(
    dataset,
    transform_train,
    transform_val,
    args,
):
    if dataset == "CIFAR100":
        dataset_train = datasets.CIFAR100(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = datasets.CIFAR100(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == "CIFAR10":
        dataset_train = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = datasets.CIFAR10(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == "MNIST":
        dataset_train = MNIST_RGB(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = MNIST_RGB(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == "FashionMNIST":
        dataset_train = FashionMNIST(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = FashionMNIST(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == "SVHN":
        dataset_train = SVHN(args.data_path, split="train", download=True, transform=transform_train)
        dataset_val = SVHN(args.data_path, split="test", download=True, transform=transform_val)

    elif dataset == "NotMNIST":
        dataset_train = NotMNIST(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = NotMNIST(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == "Flower102":
        dataset_train = Flowers102(args.data_path, split="train", download=True, transform=transform_train)
        dataset_val = Flowers102(args.data_path, split="test", download=True, transform=transform_val)

    elif dataset == "Cars196":
        dataset_train = StanfordCars(args.data_path, split="train", download=True, transform=transform_train)
        dataset_val = StanfordCars(args.data_path, split="test", download=True, transform=transform_val)

    elif dataset == "CUB200":
        dataset_train = CUB200(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = CUB200(args.data_path, train=False, download=True, transform=transform_val).data

    elif dataset == "Scene67":
        dataset_train = Scene67(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = Scene67(args.data_path, train=False, download=True, transform=transform_val).data

    elif dataset == "TinyImagenet":
        dataset_train = TinyImagenet(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = TinyImagenet(args.data_path, train=False, download=True, transform=transform_val).data

    elif dataset == "Imagenet-R":
        dataset_train = Imagenet_R(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = Imagenet_R(args.data_path, train=False, download=True, transform=transform_val).data

    # czsl datasets
    elif dataset == "ut-zappos":
        splited_dataset, mask = get_czsl_dataset("ut-zappos", transform_train, transform_val, args)
        return splited_dataset, mask

    elif dataset == "clothing16k":
        splited_dataset, mask = get_czsl_dataset("clothing16k", transform_train, transform_val, args)
        return splited_dataset, mask

    else:
        raise ValueError("Dataset {} not found.".format(dataset))

    return dataset_train, dataset_val


def split_single_dataset(dataset_train, dataset_val, args):
    nb_classes = len(dataset_val.classes)
    assert nb_classes % args.num_tasks == 0
    classes_per_task = nb_classes // args.num_tasks

    labels = [i for i in range(nb_classes)]

    split_datasets = list()
    mask = list()

    if args.shuffle:
        random.shuffle(labels)

    for _ in range(args.num_tasks):
        train_split_indices = []
        test_split_indices = []

        scope = labels[:classes_per_task]
        labels = labels[classes_per_task:]

        mask.append(scope)

        for k in range(len(dataset_train.targets)):
            if int(dataset_train.targets[k]) in scope:
                train_split_indices.append(k)

        for h in range(len(dataset_val.targets)):
            if int(dataset_val.targets[h]) in scope:
                test_split_indices.append(h)

        subset_train, subset_val = Subset(dataset_train, train_split_indices), Subset(dataset_val, test_split_indices)

        split_datasets.append([subset_train, subset_val])

    return split_datasets, mask


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3.0 / 4.0, 4.0 / 3.0)
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(args.input_size, scale=scale, ratio=ratio),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())

    return transforms.Compose(t)


def get_czsl_dataset(dataset, transform_train, transform_val, args):
    splited_dataset = []
    mask = []
    num_tasks = args.num_tasks
    dataset_root = Path(args.data_path) / dataset
    for i in range(num_tasks):
        dataset_train = Split_CZSL_Dataset(
            dataset_root,
            i,
            num_tasks,
            train=True,
            transoform=transform_train,
        )
        dataset_val = Split_CZSL_Dataset(
            dataset_root,
            i,
            num_tasks,
            train=False,
            transoform=transform_val,
        )
        splited_dataset.append([dataset_train, dataset_val])
        mask.append(utils.load_czsl_mask(dataset_root, i, num_tasks))

    return splited_dataset, mask
