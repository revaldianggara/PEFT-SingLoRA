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

    where u(t) is a ramp-up function that gradually increases from 0 to 1,
    and A* is a truncation of A for non-square cases.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        ramp_up_steps: int = 1000,
        **kwargs,
    ):
        """
        Initialize SingLoRA layer following PEFT conventions.

        Args:
            base_layer: The base layer to be adapted (e.g., nn.Linear)
            adapter_name: Name of the adapter
            r: Rank of the adaptation
            lora_alpha: LoRA scaling factor (alpha)
            lora_dropout: Dropout probability for LoRA layers
            init_lora_weights: How to initialize weights
            use_rslora: Whether to use rank-stabilized LoRA
            ramp_up_steps: Number of steps for the ramp-up function
            **kwargs: Additional arguments
        """
