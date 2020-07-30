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
        nn.Module.__init__(self)
        LoraLayer.__init__(self, base_layer, **kwargs)

        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ParameterDict({})
        self.ramp_up_steps = ramp_up_steps

        if r > 0:
            self.update_layer(
                adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora
            )

        self._disable_adapters = False
        self.merged_adapters = []

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        lora_alpha: float,
        lora_dropout: float,
        init_lora_weights: Union[bool, str],
        use_rslora: bool,
    ):
        """Update layer with new adapter."""
        # Add dropout layer
        if lora_dropout > 0.0:
            self.lora_dropout[adapter_name] = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout[adapter_name] = nn.Identity()

        # Determine dimensions based on layer type
        if isinstance(self.base_layer, nn.Linear):
            self.in_features = self.base_layer.in_features
            self.out_features = self.base_layer.out_features
        elif isinstance(self.base_layer, nn.Conv2d):
            self.in_features = self.base_layer.in_channels
            self.out_features = self.base_layer.out_channels
        else:
            raise TypeError(f"Unsupported layer type: {type(self.base_layer)}")

        # Create the single matrix A with the larger dimension
        # This follows the paper's approach where A ∈ R^(max(d_in, d_out) × r)
        self.larger_dim = max(self.in_features, self.out_features)
        self.lora_A[adapter_name] = nn.Parameter(torch.zeros(self.larger_dim, r))

        # Initialize training step counter
        self.register_buffer(f"training_step_{adapter_name}", torch.tensor(0, dtype=torch.float32))
