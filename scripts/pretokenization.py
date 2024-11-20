"""Pretokenization script for TiTok and RAR.

This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference:
    https://github.com/LTH14/mar/blob/main/main_cache.py

Example command:

torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --rdzv-endpoint=localhost:9999 \
    scripts/pretokenization.py \
    --img_size 256 \
    --batch_size 8 \
    --ten_crop \
    --data_path ${PATH_TO_IMAGENET} --cached_path ${PATH_TO_SAVE_JSONL}
"""
import builtins
import argparse
import datetime
import numpy as np
from PIL import Image
import torch.distributed as dist

import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils.train_utils import PretrainedTokenizer
import utils.misc as misc
from tqdm import tqdm
import json
import glob

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class ImageFolderWithFilename(datasets.ImageFolder):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, filename).
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        filename = path.split(os.path.sep)[-2:]
        filename = os.path.join(*filename)
        return sample, target, filename


def get_args_parser():
    parser = argparse.ArgumentParser('Cache VQ codes', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')

    # VAE parameters
    parser.add_argument('--img_size', default=256, type=int,
                        help='images input size')
    # Dataset parameters
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # caching latents
    parser.add_argument('--cached_path', default='', help='path to cached latents')

    parser.add_argument("--ten_crop", action='store_true', help="whether using random crop")

    return parser


def convert_json_to_jsonl(input_pattern, output_file):
    with open(output_file, 'w') as outfile:
        for filename in tqdm.tqdm(glob.glob(input_pattern)):
            with open(filename, 'r') as infile:
                data = json.load(infile)
                for item in data:
                    json.dump(item, outfile)
                    outfile.write('\n')


@torch.no_grad()
def main(args):
    os.makedirs(args.cached_path, exist_ok=True)
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if args.ten_crop:
        # augmentation following LLamaGen
        crop_size = int(args.img_size * 1.1)
        transform_train = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
            transforms.TenCrop(args.img_size), # this is a tuple of PIL Images
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), # returns a 4D tensor
        ])
    else:
        # augmentation following DiT and ADM
        transform_train = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # MaskGIT-VQ expects input in range of [0, 1]
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    dataset_train = ImageFolderWithFilename(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=False,
    )
    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,  # Don't drop in cache
    )
    if global_rank == 0:
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id="fun-research/TiTok",
            filename=f"maskgit-vqgan-imagenet-f16-256.bin", local_dir="./")
    if misc.is_dist_avail_and_initialized():
        torch.cuda.synchronize()

    tokenizer = PretrainedTokenizer("maskgit-vqgan-imagenet-f16-256.bin")
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    tokenizer.to(device)

    processed = []


    print(f"Start caching latents, {args.rank}, {args.gpu}")
    start_time = time.time()
    for samples, target, paths in tqdm(data_loader_train):
        samples = samples.to(device, non_blocking=True)
        
        if args.ten_crop:
            samples_all = samples.flatten(0, 1)
            target_all = target.unsqueeze(1).repeat(1, 10).flatten(0, 1)
        else:
            samples_all = torch.cat([samples, torch.flip(samples, dims=[-1])])
            target_all = torch.cat([target, target])
        with torch.no_grad():
            codes = tokenizer.encode(samples_all)

        for b in range(codes.shape[0]):
            processed.append({
                "class_id": target_all[b].cpu().item(),
                "tokens": codes[b].cpu().tolist()
            })
        
        if misc.is_dist_avail_and_initialized():
            torch.cuda.synchronize()
    print(f"{args.rank} proccessed {len(processed)} samples")
    target_json_path = f"{args.cached_path}/pretokenized_{args.rank}"
    target_json_path = target_json_path + ".json"
    with open(target_json_path, "w") as json_f:
        json.dump(processed, json_f)
    if misc.is_dist_avail_and_initialized():
        torch.cuda.synchronize()

    # write into a single jsonl
    if global_rank == 0:
        convert_json_to_jsonl(f"{args.cached_path}/pretokenized_*.json",
                              f"{args.cached_path}/pretokenized.jsonl")

    if misc.is_dist_avail_and_initialized():
        torch.cuda.synchronize()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Caching time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
