"""This file contains code to run different learning rate schedulers.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.

Reference:
    https://raw.githubusercontent.com/huggingface/open-muse/vqgan-finetuning/muse/lr_schedulers.py
"""
import math
from enum import Enum
from typing import Optional, Union

import torch


class SchedulerType(Enum):
    COSINE = "cosine"
    CONSTANT = "constant"

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    base_lr: float = 1e-4,
    end_lr: float = 0.0,
):
    """Creates a cosine learning rate schedule with warm-up and ending learning rate.

    Args:
        optimizer: A torch.optim.Optimizer, the optimizer for which to schedule the learning rate.
        num_warmup_steps: An integer, the number of steps for the warmup phase.
        num_training_steps: An integer, the total number of training steps.
        num_cycles : A float, the number of periods of the cosine function in a schedule (the default is to 
            just decrease from the max value to 0 following a half-cosine).
        last_epoch: An integer, the index of the last epoch when resuming training.
        base_lr: A float, the base learning rate.
        end_lr: A float, the final learning rate.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        ratio = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        return (end_lr + (base_lr - end_lr) * ratio) / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    base_lr: float = 1e-4,
    end_lr: float = 0.0,
):
    """UViT: Creates a constant learning rate schedule with warm-up.

    Args:
        optimizer: A torch.optim.Optimizer, the optimizer for which to schedule the learning rate.
        num_warmup_steps: An integer, the number of steps for the warmup phase.
        num_training_steps: An integer, the total number of training steps.
        num_cycles : A float, the number of periods of the cosine function in a schedule (the default is to 
            just decrease from the max value to 0 following a half-cosine).
        last_epoch: An integer, the index of the last epoch when resuming training.
        base_lr: A float, the base learning rate.
        end_lr: A float, the final learning rate.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


TYPE_TO_SCHEDULER_FUNCTION = {
    SchedulerType.COSINE: get_cosine_schedule_with_warmup,
    SchedulerType.CONSTANT: get_constant_schedule_with_warmup,
}

def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    base_lr: float = 1e-4,
    end_lr: float = 0.0,
):
    """Retrieves a learning rate scheduler from the given name and optimizer.

    Args:
        name: A string or SchedulerType, the name of the scheduler to retrieve.
        optimizer: torch.optim.Optimizer. The optimizer to use with the scheduler.
        num_warmup_steps: An integer, the number of warmup steps.
        num_training_steps: An integer, the total number of training steps.
        base_lr: A float, the base learning rate.
        end_lr: A float, the final learning rate.

    Returns:
        A instance of torch.optim.lr_scheduler.LambdaLR

    Raises:
        ValueError: If num_warmup_steps or num_training_steps is not provided.
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        base_lr=base_lr,
        end_lr=end_lr,
    )