from typing import Tuple, List, Optional, Union

import logging
import os
import pickle
import random
import signal
from copy import deepcopy
from itertools import islice
import numpy as np
import torch
import torch.distributed as dist
from torch import multiprocessing as mp

logger = logging.getLogger(__name__)

def set_all_seeds(seed: int, deterministic_cudnn: bool = False) -> None: 
    """
    Setting multiple seeds to make runs reproducible 

    Important: Enabling `deterministic_cudnn` gives you full reproducibility with CUDA 
    but might slow down your training (see https://pytorch.org/docs/stable/notes/randomness.html#cudnn)

    :param seed: number to use as seed 
    :param deterministic_torch: Enable for full reproducibility when using CUDA. Caution: might slow down training. 
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn: 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def initialize_device_settings(
    use_cuda: Optional[bool] = None, 
    local_rank: int = -1, 
    multi_gpu: bool = True, 
    devices: List[Union[str, torch.device]] = None, 
) -> Tuple[List[torch.device], int]: 
    """
    Returns a list of available devices

    :param use_cuda: Whether to make use of Cuda GPUs (is available). 
    :param local_rank: Ordinal of device to be used. If -1 and 'multi_gpu' is True, all device will be used
                        Unused if 'devices' is set or 'use_cuda' is False
    :param multi_gpu: whether to make use of all GPUs (is available). 
                        Unsued if 'devices' is set or 'use_gpu' is False. 
    :param devices: List of ...
    """
    if use_cuda is False: 
        devices_to_use = [torch.device("cpu")]
        n_gpu = 0
    elif devices: 
        if not isinstance(devices, list): 
            raise ValueError(f"devices must be a list, but got {devices} of type {type(devices)}")
        if any(isinstance(devices, str) for device in devices): 
            torch_devices: List[torch.device] = [torch.device(device) for device in devices]
            devices_to_use = torch_devices
        else: 
            devices_to_use = devices
        n_gpu = sum(1 for device in devices_to_use if "cpu" not in device.type) 
    elif local_rank == -1: 
        if torch.cuda.is_available(): 
            if multi_gpu: 
                devices_to_use = [torch.device(device) for device in range(torch.cuda.device_count())]
                n_gpu = torch.cuda.device_count()
            else: 
                devices_to_use = [torch.device("cuda")]
                n_gpu = 1
        else: 
            devices_to_use = [torch.device("cpu")]
            n_gpu = 0
    else: 
        devices_to_use = [torch.device("cuda", local_rank)]
        torch.cuda.set_device(devices_to_use[0])
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")
    logger.info(f"Using devices: {', '.join([str(device) for device in devices_to_use]).upper()}")
    logger.info(f"Number of GPUs: {n_gpu}")
    return devices_to_use, n_gpu
