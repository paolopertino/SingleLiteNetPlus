import torch


def save_checkpoint(state, filenameCheckpoint="checkpoint.pth.tar"):
    torch.save(state, filenameCheckpoint)
