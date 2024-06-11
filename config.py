DATA_ROOT = "/work/u8083200/Thesis/datasets/SingleHDR_training_data"

WEIGHT_NAME = "MEGHDR"
# Training
EPOCH = 61
BATCH_SIZE = 6

AUG = True

EXPOSURE_TIME = {"t_1": 1.0, "t_2": 2.0}

loss_weight = {
    "representation_loss": 1e0,
    "reconstruction_loss": 1e2,
    "perceptual_loss": 1e-2,
    "tv_loss": 5e-4,
    }
    
RESULT_SAVE_PATH = "/work/u8083200/Thesis/SOTA/MEGHDR/result"
WEIGHT_SAVE_PATH = "/work/u8083200/Thesis/SOTA/MEGHDR/weight"


LEARNING_RATE = 1e-4
DEVICE = "cuda"

import torch
import random
import numpy as np
def set_random_seed(seed):
    # Set the random seed for Python RNG
    random.seed(seed)
    np.random.seed(seed)

    # Set the random seed for PyTorch RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False