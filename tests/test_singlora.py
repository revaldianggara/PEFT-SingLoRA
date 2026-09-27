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
        lora_alpha=16,
    )

    # Check that layer was created
    assert isinstance(singlora_layer, Linear)
    assert singlora_layer.r == 4
    assert singlora_layer.lora_alpha == 16


def test_singlora_forward_pass():
    """Test forward pass through SingLoRA layer."""
    base_layer = nn.Linear(10, 20)
    singlora_layer = Linear(
        base_layer=base_layer,
        adapter_name="default",
        r=4,
        lora_alpha=16,
    )

    # Test forward pass
    x = torch.randn(5, 10)
    output = singlora_layer(x)

    assert output.shape == (5, 20)
    assert torch.isfinite(output).all()


def test_peft_integration():
    """Test integration with PEFT."""
    setup_singlora()

    model = SimpleModel()
    config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["linear1", "linear2"],
        lora_dropout=0.1,
    )

    peft_model = get_peft_model(model, config)

    # Check that model was created
    assert peft_model is not None

    # Test forward pass
    x = torch.randn(2, 10)
    output = peft_model(x)
    assert output.shape == (2, 5)


def test_singlora_config():
    """Test SingLoRAConfig."""
    config = SingLoRAConfig(
        r=8,
        lora_alpha=32,
        target_modules=["linear1"],
        ramp_up_steps=2000,
    )

    assert config.r == 8
    assert config.lora_alpha == 32
    assert config.ramp_up_steps == 2000


def test_parameter_count():
    """Test that SingLoRA has fewer parameters than standard LoRA."""
    model = SimpleModel()

    # Setup SingLoRA
    setup_singlora()
    config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["linear1", "linear2"],
    )

    peft_model = get_peft_model(model, config)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())

    # SingLoRA should have fewer trainable params than standard LoRA
    # For two Linear layers:
    # - linear1 (10→20): A matrix is (20, 4) = 80 params
    # - linear2 (20→5): A matrix is (20, 4) = 80 params
    # Total: 160 trainable params
    # This is about 33% of total params, not < 10%

    # Standard LoRA would have:
    # - linear1: A(10,4) + B(4,20) = 40 + 80 = 120 params
    # - linear2: A(20,4) + B(4,5) = 80 + 20 = 100 params
    # Total: 220 params

    # So SingLoRA has 160/220 ≈ 73% of standard LoRA params
    assert trainable_params == 160
    assert trainable_params < 200  # Less than what standard LoRA would have


def test_merge_unmerge():
    """Test merging and unmerging of adapters."""
    setup_singlora()

    model = SimpleModel()
    config = SingLoRAConfig(
        r=4,
        lora_alpha=16,
        target_modules=["linear1"],
        # Use small ramp_up_steps so we get non-zero update immediately
        ramp_up_steps=1,
    )

    peft_model = get_peft_model(model, config)

    # Use update_singlora_global_step to increment training step
    update_singlora_global_step(peft_model, 1)

    # Access the SingLoRA layer
    singlora_layer = peft_model.base_model.model.linear1

    # Get original weight
    original_weight = peft_model.base_model.model.linear1.base_layer.weight.clone()

    # Merge adapter
    peft_model.merge_adapter()
    merged_weight = peft_model.base_model.model.linear1.base_layer.weight.clone()

    # Weights should be different after merge
    assert not torch.allclose(original_weight, merged_weight)

    # Unmerge
    peft_model.unmerge_adapter()
    unmerged_weight = peft_model.base_model.model.linear1.base_layer.weight.clone()

    # Should be back to original
    assert torch.allclose(original_weight, unmerged_weight, atol=1e-6)


def test_ramp_up_function():
    """Test that the ramp-up function works correctly."""
    setup_singlora()

    base_layer = nn.Linear(10, 10)
    singlora_layer = Linear(
        base_layer=base_layer,
        adapter_name="default",
        r=4,
        lora_alpha=16,
        ramp_up_steps=100,
    )

    # Initially, update should be zero
    update_weight_0 = singlora_layer._get_update_weight("default")
    assert torch.allclose(update_weight_0, torch.zeros_like(update_weight_0))

    # After 50 steps, should be scaled by 0.5
    update_singlora_global_step(singlora_layer, 50)
    update_weight_50 = singlora_layer._get_update_weight("default")

    # After 100 steps, should be fully scaled
    update_singlora_global_step(singlora_layer, 100)
    update_weight_100 = singlora_layer._get_update_weight("default")

    # Check that magnitude increases
    assert torch.norm(update_weight_50) > torch.norm(update_weight_0)
    assert torch.norm(update_weight_100) > torch.norm(update_weight_50)

    # Check that 50-step update is half of 100-step update
    assert torch.allclose(update_weight_50, 0.5 * update_weight_100, atol=1e-6)


def test_merge_matches_reference():
    """Merged weights must equal base + scaling * A @ A.T at full ramp-up."""
    base_layer = nn.Linear(10, 10, bias=False)
    singlora_layer = Linear(
        base_layer=base_layer,
        adapter_name="default",
        r=2,
        lora_alpha=4,
        ramp_up_steps=10,
    )
    with torch.no_grad():
        A = torch.arange(20, dtype=torch.float32).reshape(10, 2) / 10
        singlora_layer.lora_A["default"].copy_(A)
    update_singlora_global_step(singlora_layer, 10)

    original = base_layer.weight.data.clone()
    expected = singlora_layer.scaling["default"] * (A @ A.T)
    singlora_layer.merge()

    assert torch.allclose(base_layer.weight.data, original + expected, atol=1e-6)


def test_merge_is_dtype_safe():
    """Merging into a half-precision base layer must not raise a dtype error."""
    base_layer = nn.Linear(8, 8, bias=False).half()
    singlora_layer = Linear(
        base_layer=base_layer,
        adapter_name="default",
        r=2,
        lora_alpha=2,
        ramp_up_steps=1,
    )
    update_singlora_global_step(singlora_layer, 1)

    singlora_layer.merge()

    assert base_layer.weight.data.dtype == torch.float16
    assert torch.isfinite(base_layer.weight.data).all()


def test_safe_merge_rejects_nonfinite():
    """A safe merge must refuse an update that would poison the weights."""
    base_layer = nn.Linear(6, 6, bias=False)
    singlora_layer = Linear(
        base_layer=base_layer,
        adapter_name="default",
        r=2,
        lora_alpha=2,
        ramp_up_steps=1,
    )
    with torch.no_grad():
        singlora_layer.lora_A["default"].fill_(float("inf"))
    update_singlora_global_step(singlora_layer, 1)

    with pytest.raises(ValueError):
        singlora_layer.merge(safe_merge=True)

    assert singlora_layer.merged_adapters == []


def test_non_square_update_shapes():
    """The update matrix must always come out as (out_features, in_features)."""
    for in_features, out_features in ((10, 20), (20, 10)):
        base_layer = nn.Linear(in_features, out_features, bias=False)
        singlora_layer = Linear(
            base_layer=base_layer,
            adapter_name="default",
            r=3,
            lora_alpha=3,
            ramp_up_steps=1,
        )
        with torch.no_grad():
            singlora_layer.lora_A["default"].normal_(std=0.1)
        update_singlora_global_step(singlora_layer, 1)

        update = singlora_layer._get_update_weight("default")
        assert update.shape == (out_features, in_features)

        A = singlora_layer.lora_A["default"]
        smaller = min(in_features, out_features)
        A_star = A[:smaller, :]
        if in_features < out_features:
            reference = A @ A_star.T
        else:
            reference = A_star @ A.T
        assert torch.allclose(update, singlora_layer.scaling["default"] * reference, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
