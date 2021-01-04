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
