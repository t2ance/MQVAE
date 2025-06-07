import io
import time
from copy import deepcopy

import albumentations
import numpy as np
import torch
from PIL import Image
from datasets import load_dataset, IterableDataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import Dataset

from arguments import TrainingArguments
from utils import wrap_with_loader


class ImageNetDatasetFromHuggingFace(Dataset):
    def __init__(
            self,
            hf_dataset,
            image_size
    ):
        self.image_size = image_size
        self.dataset = hf_dataset
        self.preprocessor = albumentations.Compose([
            albumentations.SmallestMaxSize(max_size=image_size),
            albumentations.RandomCrop(height=image_size, width=image_size),
            albumentations.HorizontalFlip(p=0.5)
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.map(self.dataset[index]).values()

    def map(self, example):
        label = example['label']
        image = example['image']

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        if image.mode != "RGB":
            image = image.convert("RGB")
        image_np = np.array(image).astype(np.uint8)
        image_processed = self.preprocessor(image=image_np)["image"]
        image_processed = (image_processed / 127.5 - 1.0).astype(np.float32)
        image_tensor = torch.from_numpy(image_processed).permute(2, 0, 1)

        example['image'], example['label'] = image_tensor, torch.tensor(label)
        return example


class FFHQDatasetFromHuggingFace(Dataset):
    def __init__(
            self,
            hf_dataset,
            image_size: int
    ):
        self.image_size = image_size
        self.dataset = hf_dataset
        self.rescaler = albumentations.SmallestMaxSize(max_size=image_size)
        self.cropper = albumentations.RandomCrop(height=image_size, width=image_size)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.map(self.dataset[index]).values()

    def map(self, example):
        image = example['image']
        if isinstance(image, dict) and 'bytes' in image:
            image = Image.open(io.BytesIO(image['bytes']))
        elif isinstance(image, Image.Image):
            pass
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, str):
            image = Image.open(image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        if not image.mode == "RGB":
            image = image.convert("RGB")

        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image)

        example['image'], example['label'] = image, torch.tensor(0)
        return example


class CIFAR10DatasetFromHuggingFace(Dataset):
    def __init__(self, image_size):
        self.image_size = image_size

        # CIFAR10 contains 10 classes of 32x32 RGB images.
        # We use the default train/test split (50,000/10,000 samples) and
        # further split 10,000 samples from the training set as the validation set.

    def map(self, example):
        # Convert image to PIL Image if not already
        if not isinstance(example['img'], Image.Image):
            example['img'] = Image.fromarray(example['img'])

        # Ensure the image is in RGB format
        if example['img'].mode != "RGB":
            example['img'] = example['img'].convert("RGB")

        # Convert image to numpy array and normalize
        image = np.array(example['img']).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = np.transpose(image, (2, 0, 1))  # Convert to CHW format

        return {
            'image': torch.from_numpy(image),
            'label': torch.tensor(example['label'])
        }


class CelebADatasetFromHuggingFace:
    def __init__(self, image_size, seed=None):

        self.preprocessor = albumentations.Compose([
            albumentations.RandomCrop(height=140, width=140),
            albumentations.SmallestMaxSize(max_size=image_size)
        ], seed=seed)
        print(f'Using preprocessor {self.preprocessor}')

    def map(self, example):
        image = example['image']
        label = 0

        if not isinstance(image, Image.Image):
            if isinstance(image, dict):
                print(f'image keys: {list(image.keys())}')
            image = Image.fromarray(image)

        # Ensure image is in RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Convert image to numpy array
        image_np = np.array(image).astype(np.uint8)

        # Apply preprocessing
        image_processed = self.preprocessor(image=image_np)["image"]

        # Normalize the image
        image_processed = (image_processed / 127.5 - 1.0).astype(np.float32)

        # Permute dimensions to CHW
        image_tensor = torch.from_numpy(image_processed).permute(2, 0, 1)

        # Convert label to tensor
        label_tensor = torch.tensor(label).long()  # Use long for integer labels

        # Return as a dictionary
        return {
            'image': image_tensor,
            'label': label_tensor
        }


def get_debugger():
    from icecream import ic
    from datetime import datetime, timezone, timedelta

    ic.configureOutput(includeContext=True, prefix=lambda: f'{datetime.now(timezone(timedelta(hours=8)))}|> ')
    return ic


def train_test_split_for_stream(dataset, test_size):
    def is_train(example, hash_mod=100):
        sample_string = str(example)
        hash_value = hash(sample_string)
        return (hash_value % hash_mod) >= (test_size * hash_mod)

    def is_test(example, hash_mod=100):
        sample_string = str(example)
        hash_value = hash(sample_string)
        return (hash_value % hash_mod) < (test_size * hash_mod)

    train_dataset = dataset.filter(is_train)
    test_dataset = dataset.filter(is_test)

    return train_dataset, test_dataset


def load_dataset_for_training(args: TrainingArguments):
    if args.dataset == "imagenet":
        dataset_path = "ILSVRC/imagenet-1k"
        map_func_train = ImageNetDatasetFromHuggingFace(hf_dataset=None, image_size=args.image_size).map
        map_func_val = ImageNetDatasetFromHuggingFace(hf_dataset=None, image_size=args.image_size).map
        split_train = "train"
        split_val = "validation"
        remove_columns = None
    elif args.dataset == "ffhq":
        dataset_path = "bitmind/ffhq-256"
        map_func_train = FFHQDatasetFromHuggingFace(hf_dataset=None, image_size=args.image_size).map
        map_func_val = FFHQDatasetFromHuggingFace(hf_dataset=None, image_size=args.image_size).map
        split_train = "train"
        split_val = "train"
        remove_columns = None
    elif args.dataset == "cifar10":
        dataset_path = "uoft-cs/cifar10"
        map_func_train = CIFAR10DatasetFromHuggingFace(image_size=args.image_size).map
        map_func_val = CIFAR10DatasetFromHuggingFace(image_size=args.image_size).map
        split_train = "train"
        split_val = "test"
        remove_columns = ['img']
    elif args.dataset == 'celeba':
        dataset_path = "flwrlabs/celeba"
        map_func_train = CelebADatasetFromHuggingFace(image_size=args.image_size, seed=args.seed).map
        map_func_val = CelebADatasetFromHuggingFace(image_size=args.image_size, seed=args.seed).map
        split_train = "train"
        remove_columns = None
    else:
        raise NotImplementedError()

    args.log(f'Loading dataset from data_path {dataset_path}')
    begin = time.time()
    dataset_train = load_dataset(
        dataset_path,
        cache_dir=None if args.streaming else args.cache_dir,
        trust_remote_code=True,
        split=split_train,
        token=args.hf_token,
        streaming=args.streaming,
        num_proc=None if args.streaming else args.num_workers
    )
    dataset_val = load_dataset(
        dataset_path,
        cache_dir=None if args.streaming else args.cache_dir,
        trust_remote_code=True,
        split=split_val,
        token=args.hf_token,
        streaming=args.streaming,
        num_proc=None if args.streaming else args.num_workers
    )
    end = time.time()
    args.log(f'loading dataset done: duration: {end - begin}s')

    if args.streaming:
        dataset_train = dataset_train.map(map_func_train, remove_columns=remove_columns)
        dataset_val = dataset_val.map(map_func_val, remove_columns=remove_columns)
        args.log(f'n_shards of dataset_train :{dataset_train.n_shards}')
        args.log(f'n_shards of dataset_val :{dataset_val.n_shards}')
    else:
        dataset_train = dataset_train.map(
            map_func_train,
            num_proc=args.num_workers,
            remove_columns=remove_columns,
            load_from_cache_file=False
        )
        dataset_val = dataset_val.map(
            map_func_val,
            num_proc=args.num_workers,
            remove_columns=remove_columns,
            load_from_cache_file=False
        )
        dataset_train.set_format(type='pt', columns=['image', 'label'])
        dataset_val.set_format(type='pt', columns=['image', 'label'])

    if args.max_train_samples > 0:
        if isinstance(dataset_train, IterableDataset):
            dataset_train = dataset_train.take(args.max_train_samples)
        else:
            dataset_train = dataset_train[:args.max_train_samples]
        args.log(f'Truncating training dataset to {args.max_train_samples} samples')

    if args.max_val_samples > 0:
        if isinstance(dataset_val, IterableDataset):
            dataset_val = dataset_val.take(args.max_val_samples)
        else:
            dataset_val = dataset_val[:args.max_val_samples]
        args.log(f'Truncating validation dataset to {args.max_val_samples} samples')

    dataset_train_inner, dataset_train_outer = deepcopy(dataset_train), deepcopy(dataset_train)

    if args.is_distributed() and args.streaming:
        dataset_train = split_dataset_by_node(
            dataset_train, rank=args.local_rank, world_size=args.world_size
        ).shuffle(args.seed)
        dataset_val = split_dataset_by_node(
            dataset_val, rank=args.local_rank, world_size=args.world_size
        ).shuffle(args.seed)
        dataset_train_inner = split_dataset_by_node(
            dataset_train_inner, rank=args.local_rank, world_size=args.world_size
        ).shuffle(args.seed)
        dataset_train_outer = split_dataset_by_node(
            dataset_train_outer, rank=args.local_rank, world_size=args.world_size
        ).shuffle(args.seed)

    batch = next(iter(wrap_with_loader(dataset_train, args, 2)))["image"]
    args.log(f'Image shape {batch.shape}')

    return {
        'dataset_train': dataset_train,
        'dataset_train_inner': dataset_train_inner,
        'dataset_train_outer': dataset_train_outer,
        'dataset_val': dataset_val
    }


if __name__ == '__main__':
    ...
