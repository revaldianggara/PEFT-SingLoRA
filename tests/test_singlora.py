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
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


def test_singlora_layer_creation():
    """Test that SingLoRA layer can be created."""
    base_layer = nn.Linear(10, 20)
    singlora_layer = Linear(
        base_layer=base_layer,
        adapter_name="default",
        r=4,
