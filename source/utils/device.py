# import torch

# def get_device():
#     return "mps" if torch.backends.mps.is_available() else "cpu"

import torch
import platform

def get_device():
    system = platform.system()

    if system == "Darwin":  # macOS
        return "mps" if torch.backends.mps.is_available() else "cpu"
    elif system == "Windows" or system == "Linux":
        return "cuda" if torch.cuda.is_available() else "cpu"
    else:
        return "cpu"