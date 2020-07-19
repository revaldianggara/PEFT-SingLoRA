import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Union, List
from peft.tuners.lora.layer import LoraLayer


def update_singlora_global_step(model: nn.Module, global_step: int):
    """
    Update the global step for all SingLoRA layers in the model.

    Args:
        model: The model containing SingLoRA layers.
        global_step: The current global step to set.
    """
    for name, layer in model.named_modules():
        if isinstance(layer, SingLoRALayer):
            layer.update_global_step(global_step)


class SingLoRALayer(nn.Module, LoraLayer):
    """
    This layer implements the SingLoRA approach using a single matrix 'A' instead of
    the typical LoRA's two matrices. The weight update is calculated as:
    W = W_0 + alpha/r * u(t) * A @ A.T (for square matrices)
    W = W_0 + alpha/r * u(t) * A* @ A.T (for non-square matrices)
