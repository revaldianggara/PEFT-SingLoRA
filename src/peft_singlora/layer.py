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

        # Initialize weights
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        # Set scaling
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

    def reset_lora_parameters(self, adapter_name: str, init_lora_weights: Union[bool, str]):
        """Reset/initialize the LoRA parameters."""
        if adapter_name in self.lora_A.keys():
            if init_lora_weights is True:
                # Kaiming uniform initialization as in the original SingLoRA
                nn.init.kaiming_uniform_(self.lora_A[adapter_name], a=math.sqrt(5))
            elif init_lora_weights.lower() == "gaussian":
                nn.init.normal_(self.lora_A[adapter_name], std=1 / self.r[adapter_name])
            else:
                raise ValueError(f"Unknown init method: {init_lora_weights}")

    def _get_update_weight(self, adapter_name: str) -> torch.Tensor:
        """Calculate the low-rank weight update for a specific adapter."""
        # Get training step for this adapter
        training_step = getattr(self, f"training_step_{adapter_name}")

        # Ramp-up function u(t) = min(t/T, 1)
        ramp_up_factor = torch.min(
            training_step / self.ramp_up_steps,
            torch.tensor(1.0, device=training_step.device),
        ).item()

        # Get the A matrix for this adapter
        A = self.lora_A[adapter_name]  # Shape: (larger_dim, r)

        # Handle different matrix shapes according to the paper
        if self.in_features == self.out_features:
            # Square matrix: use standard A @ A.T
            update = A @ A.T
        else:
            # Non-square matrix: use A* @ A.T where A* is truncated A
            smaller_dim = min(self.in_features, self.out_features)
            A_star = A[:smaller_dim, :]  # Shape: (smaller_dim, r)

            if self.in_features < self.out_features:
                # Case from paper: d_in < d_out
                # A has shape (d_out, r), A* has shape (d_in, r)
                # A @ A*.T has shape (d_out, d_in) = correct weight shape
                update = A @ A_star.T
            else:
                # Transposed case: d_in > d_out
                # A has shape (d_in, r), A* has shape (d_out, r)
                # We need shape (d_out, d_in), so we compute A* @ A.T
                update = A_star @ A.T

        # Apply scaling and ramp-up
        return ramp_up_factor * self.scaling[adapter_name] * update

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass with SingLoRA adaptation."""
        # Handle base layer forward
        result = self.base_layer(x, *args, **kwargs)

        if self.disable_adapters:
            return result

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            # Apply dropout
            dropout = self.lora_dropout[active_adapter]

            # Get the weight update
            update_weight = self._get_update_weight(active_adapter)

            # Apply the update based on layer type
            if isinstance(self.base_layer, nn.Linear):
                # For linear layers: y = x @ W.T + b
                # With update: y = x @ (W + ΔW).T + b = x @ W.T + x @ ΔW.T + b
                result = result + F.linear(dropout(x), update_weight, None)
            elif isinstance(self.base_layer, nn.Conv2d):
                # For conv layers, we need to reshape the update appropriately
                # This is a simplified version - full implementation would need proper handling
                raise NotImplementedError("Conv2d support needs proper implementation")

        return result

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """Merge adapter weights into the base layer."""
        if adapter_names is None:
            adapter_names = self.active_adapters

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                if safe_merge:
                    # Check that we can safely merge
                    orig_dtype = self.base_layer.weight.data.dtype
                    update = self._get_update_weight(active_adapter)
                    update = update.to(orig_dtype)

                    # Test merge doesn't cause issues
                    new_weight = self.base_layer.weight.data + update
                    if not torch.isfinite(new_weight).all():
                        raise ValueError(
                            f"Merging adapter {active_adapter} would result in NaN/Inf"
                        )

                # Perform the merge
                self.base_layer.weight.data += self._get_update_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """Unmerge adapter weights from the base layer."""
        while self.merged_adapters:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                self.base_layer.weight.data -= self._get_update_weight(active_adapter)

    def update_global_step(self, global_step: int):
        """Update global step for all adapters."""
        for adapter_name in self.lora_A.keys():
            step_buffer = getattr(self, f"training_step_{adapter_name}")
            step_buffer.fill_(global_step)

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "singlora." + rep


class Linear(SingLoRALayer):
    """SingLoRA adapter for nn.Linear layers."""

    def __init__(
        self,
        base_layer: nn.Linear,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        **kwargs,
    ):
        """
        Initialize SingLoRA adapter for Linear layer.

        Args:
            base_layer: The nn.Linear layer to adapt
            adapter_name: Name of the adapter
            r: Rank of the adaptation
            lora_alpha: Scaling factor
            lora_dropout: Dropout probability
            fan_in_fan_out: Set to True if the layer stores weight like (fan_in, fan_out)
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(
            base_layer,
            adapter_name,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            **kwargs,
        )

# draft note 903
