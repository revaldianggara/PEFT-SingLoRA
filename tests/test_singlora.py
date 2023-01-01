"""
Basic tests for SingLoRA implementation.
"""

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

# Import after peft to avoid issues
from peft_singlora import setup_singlora, Linear, SingLoRAConfig, update_singlora_global_step


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
