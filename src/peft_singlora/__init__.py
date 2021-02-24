"""
PEFT-SingLoRA: Single Low-Rank Adaptation for PEFT

A parameter-efficient fine-tuning method that uses a single low-rank matrix
instead of two, reducing parameters while maintaining performance.
"""

from .layer import SingLoRALayer, Linear, update_singlora_global_step
from .config import SingLoRAConfig, setup_singlora
