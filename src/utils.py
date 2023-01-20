import torch
from pytorch_lightning import seed_everything


def set_seed(seed: int = 0):
    seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
