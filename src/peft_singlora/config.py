"""
Configuration utilities for SingLoRA.
"""

from typing import Optional, Union, List
from peft import LoraConfig


class SingLoRAConfig(LoraConfig):
    """
    Configuration class for SingLoRA with additional parameters.

    Inherits from LoraConfig to maintain compatibility while adding
    SingLoRA-specific parameters.
    """

    def __init__(
        self,
        r: int = 8,
        lora_alpha: int = 8,
        target_modules: Optional[Union[List[str], str]] = None,
        lora_dropout: float = 0.0,
        ramp_up_steps: int = 1000,
        **kwargs,
    ):
        """
        Initialize SingLoRA configuration.

        Args:
            r: Rank of adaptation
            lora_alpha: Scaling factor
            target_modules: Modules to apply SingLoRA to
            lora_dropout: Dropout probability
            ramp_up_steps: Number of steps for ramp-up function
            **kwargs: Additional LoraConfig parameters
        """
        super().__init__(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            **kwargs,
        )
        self.ramp_up_steps = ramp_up_steps

        # Auto-register SingLoRA modules
        from .layer import Linear
        import torch.nn as nn

        custom_module_mapping = {nn.Linear: Linear}
        self._register_custom_module(custom_module_mapping)


def setup_singlora():
    """
    Globally configure PEFT to use SingLoRA for Linear layers.

    This function monkey-patches LoraConfig to automatically use
    SingLoRA layers instead of standard LoRA layers.
    """
    import warnings

    try:
        from peft import LoraConfig
        import torch.nn as nn
        from .layer import Linear

        # Store original init
        if not hasattr(LoraConfig, "_original_init"):
            LoraConfig._original_init = LoraConfig.__init__

        def _patched_init(self, *args, **kwargs):
            # Extract ramp_up_steps if provided
            ramp_up_steps = kwargs.pop("ramp_up_steps", 1000)

            # Call original init
            LoraConfig._original_init(self, *args, **kwargs)

            # Register SingLoRA modules
            custom_module_mapping = {nn.Linear: Linear}
            self._register_custom_module(custom_module_mapping)

            # Store ramp_up_steps for later use
            self.ramp_up_steps = ramp_up_steps

        LoraConfig.__init__ = _patched_init

        print("✓ SingLoRA successfully registered with PEFT")

    except ImportError:
        warnings.warn("PEFT not installed. Please install it with: pip install peft>=0.7.0")

# draft note 877
