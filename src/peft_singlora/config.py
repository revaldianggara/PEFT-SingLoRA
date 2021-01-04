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
