"""This file contains the definition of data loader using webdataset.

This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference:
    https://github.com/mlfoundations/open_clip/blob/main/src/training/data.py
    https://github.com/huggingface/open-muse/blob/main/training/data.py
"""

import math
from typing import List, Union, Text
import webdataset as wds
import torch
from torch.utils.data import default_collate
from torchvision import transforms
from torch.utils.data import Dataset
import linecache
import json


def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f


class ImageTransform:
    def __init__(self,
                 resize_shorter_edge: int = 256,
                 crop_size: int = 256,
                 random_crop: bool = True,
                 random_flip: bool = True,
                 normalize_mean: List[float] = [0., 0., 0.],
                 normalize_std: List[float] = [1., 1., 1.]):
        """Initializes the WebDatasetReader with specified augmentation parameters.

        Args:
            resize_shorter_edge: An integer, the shorter edge size to resize the input image to.
            crop_size: An integer, the size to crop the input image to.
            random_crop: A boolean, whether to use random crop augmentation during training.
            random_flip: A boolean, whether to use random flipping augmentation during training.
            normalize_mean: A list of float, the normalization mean used to normalize the image tensor.
            normalize_std: A list of float, the normalization std used to normalize the image tensor.
        
        Raises:
            NotImplementedError: If the interpolation mode is not one of ["bicubic", "bilinear"].
        """
        train_transform = []
        interpolation = transforms.InterpolationMode.BICUBIC

        train_transform.append(
            transforms.Resize(resize_shorter_edge, interpolation=interpolation, antialias=True))
        if random_crop:
            train_transform.append(transforms.RandomCrop(crop_size))
        else:
            train_transform.append(transforms.CenterCrop(crop_size))
        if random_flip:
            train_transform.append(transforms.RandomHorizontalFlip())
        train_transform.append(transforms.ToTensor())
        # normalize_mean = [0, 0, 0] and normalize_std = [1, 1, 1] will normalize images into [0, 1],
        # normalize_mean = [0.5, 0.5, 0.5] and normalize_std = [0.5, 0.5, 0.5] will normalize images into [-1, 1].
        train_transform.append(transforms.Normalize(normalize_mean, normalize_std))

        self.train_transform = transforms.Compose(train_transform)
        self.eval_transform = transforms.Compose(
            [
                # Note that we always resize to crop_size during eval to ensure the results
                # can be compared against reference numbers on ImageNet etc.
                transforms.Resize(crop_size, interpolation=interpolation, antialias=True),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std)
            ]
        )
        print(f"self.train_transform: {self.train_transform}")
        print(f"self.eval_transform: {self.eval_transform}")


class SimpleImageDataset:
    def __init__(
        self,
        train_shards_path: Union[Text, List[Text]],
        eval_shards_path: Union[Text, List[Text]],
        num_train_examples: int,
        per_gpu_batch_size: int,
        global_batch_size: int,
        num_workers_per_gpu: int = 12,
        resize_shorter_edge: int = 256,
        crop_size: int = 256,
        random_crop = True,
        random_flip = True,
        normalize_mean: List[float] = [0., 0., 0.],
        normalize_std: List[float] = [1., 1., 1.],
    ):
        """Initializes the WebDatasetReader class.

        Args:
            train_shards_path: A string or list of string, path to the training data shards in webdataset format.
            eval_shards_path: A string or list of string, path to the evaluation data shards in webdataset format.
            num_train_examples: An integer, total number of training examples.
            per_gpu_batch_size: An integer, number of examples per GPU batch.
            global_batch_size: An integer, total number of examples in a batch across all GPUs.
            num_workers_per_gpu: An integer, number of workers per GPU.
            resize_shorter_edge: An integer, the shorter edge size to resize the input image to.
            crop_size: An integer, the size to crop the input image to.
            random_crop: A boolean, whether to use random crop augmentation during training.
            random_flip: A boolean, whether to use random flipping augmentation during training.
            normalize_mean: A list of float, the normalization mean used to normalize the image tensor.
            normalize_std: A list of float, the normalization std used to normalize the image tensor.
        """
        transform = ImageTransform(
            resize_shorter_edge, crop_size, random_crop, random_flip,
            normalize_mean, normalize_std)

        train_processing_pipeline = [
            wds.decode(wds.autodecode.ImageHandler("pil", extensions=["webp", "png", "jpg", "jpeg"])),
            wds.rename(
                image="jpg;png;jpeg;webp",
                class_id="cls",
                handler=wds.warn_and_continue,
                ),
            wds.map(filter_keys(set(["image", "class_id", "filename"]))),
            wds.map_dict(
                image=transform.train_transform,
                class_id=lambda x: int(x),
                handler=wds.warn_and_continue,
            ),
        ]

        test_processing_pipeline = [
            wds.decode(wds.autodecode.ImageHandler("pil", extensions=["webp", "png", "jpg", "jpeg"])),
            wds.rename(
                image="jpg;png;jpeg;webp",
                class_id="cls",
                handler=wds.warn_and_continue,
                ),
            wds.map(filter_keys(set(["image", "class_id", "filename"]))),
            wds.map_dict(
                image=transform.eval_transform,
                class_id=lambda x: int(x),
                handler=wds.warn_and_continue,
            ),
        ]

        # Create train dataset and loader.
        pipeline = [
            wds.ResampledShards(train_shards_path),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(bufsize=5000,
                        initial=1000),
            *train_processing_pipeline,
            wds.batched(per_gpu_batch_size, partial=False, collation_fn=default_collate),
        ]

        num_batches = math.ceil(num_train_examples / global_batch_size)
        num_worker_batches = math.ceil(num_train_examples / 
            (global_batch_size * num_workers_per_gpu))
        num_batches = num_worker_batches * num_workers_per_gpu
        num_samples = num_batches * global_batch_size

        # Each worker is iterating over the complete dataset.
        self._train_dataset = wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)
        self._train_dataloader = wds.WebLoader(
            self._train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers_per_gpu,
            pin_memory=True,
            persistent_workers=True,
        )
        # Add meta-data to dataloader instance for convenience.
        self._train_dataloader.num_batches = num_batches
        self._train_dataloader.num_samples = num_samples

        # Create eval dataset and loader.
        pipeline = [
            wds.SimpleShardList(eval_shards_path),
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.ignore_and_continue),
            *test_processing_pipeline,
            wds.batched(per_gpu_batch_size, partial=True, collation_fn=default_collate),
        ]
        self._eval_dataset = wds.DataPipeline(*pipeline)
        self._eval_dataloader = wds.WebLoader(
            self._eval_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers_per_gpu,
            pin_memory=True,
            persistent_workers=True,
        )

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def train_dataloader(self):
        return self._train_dataloader

    @property
    def eval_dataset(self):
        return self._eval_dataset

    @property
    def eval_dataloader(self):
        return self._eval_dataloader
    

class PretoeknizedDataSetJSONL(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.jsonl_file = data_path
        self.num_lines = sum(1 for _ in open(self.jsonl_file))
        # Ensure the file is cached
        linecache.checkcache(self.jsonl_file)
        print("Number of data:", self.num_lines)

    def __len__(self):
        return self.num_lines

    def __getitem__(self, idx):
        line = linecache.getline(self.jsonl_file, idx + 1).strip()
        data = json.loads(line)
        return torch.tensor(data["class_id"]), torch.tensor(data["tokens"])