# PEFT-SingLoRA

[![Tests](https://github.com/revaldianggara/PEFT-SingLoRA/actions/workflows/tests.yml/badge.svg)](https://github.com/revaldianggara/PEFT-SingLoRA/actions/workflows/tests.yml)

[![PyPI version](https://badge.fury.io/py/peft-singlora.svg)](https://badge.fury.io/py/peft-singlora)
[![License](https://img.shields.io/badge/License-BSD_2--Clause-blue.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

SingLoRA (Single Low-Rank Adaptation) is an efficient alternative to traditional LoRA that uses a single low-rank matrix instead of two, reducing parameters while maintaining performance. This package provides a PEFT-compatible implementation of SingLoRA for the Hugging Face ecosystem: register it once and `get_peft_model` will use it for the targeted linear layers.

## Key Features

- 🚀 **50% fewer parameters** than standard LoRA
- 🔧 **Fully compatible with PEFT** ecosystem
- 📊 **Mathematically equivalent** to standard LoRA
- 🎯 **Easy integration** with existing PEFT workflows

## Installation

```bash
pip install peft-singlora
```

## Quick Start

Here is an extremely simplified training loop example to help understand when to update the global step:

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from peft_singlora import setup_singlora, update_singlora_global_step

# Load your model
model = AutoModelForCausalLM.from_pretrained("your-model-name")

# Setup SingLoRA (this registers it with PEFT)
setup_singlora()

# Configure LoRA as usual - it will use SingLoRA under the hood
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
)

# Create PEFT model - will automatically use SingLoRA for linear layers
peft_model = get_peft_model(model, config)

# Training loop with proper step tracking
gradient_accumulation_steps = 4
global_step = 0

for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(dataloader):
        # Forward pass
        outputs = peft_model(**batch)
        loss = outputs.loss / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights and global step every N batches
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            # Update SingLoRA step counter after optimizer step
            update_singlora_global_step(peft_model, global_step)
            global_step += 1
```

## How It Works

Traditional LoRA uses two matrices (A and B) for the low-rank decomposition:
```
W = W_0 + BA
```

SingLoRA uses a single matrix A with a symmetric decomposition:
```
W = W_0 + α/r * A @ A^T
```

This reduces trainable parameters from `2 * d * r` to `d * r` while maintaining the same expressive power.

## Advanced Usage

### Custom Configuration

```python
from peft_singlora import SingLoRAConfig

config = SingLoRAConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj"],
    lora_dropout=0.1,
    ramp_up_steps=1000,  # Gradually increase adaptation strength
)
```

### Manual Integration

```python
import torch.nn as nn
from peft_singlora import Linear as SingLoRALinear

# Register custom module mapping
custom_module_mapping = {nn.Linear: SingLoRALinear}
config._register_custom_module(custom_module_mapping)
```

## Examples

Check out the [examples/](https://github.com/revaldianggara/PEFT-SingLoRA/tree/main/examples) directory for:
- Basic usage with different model architectures
- Fine-tuning examples with real datasets
- Performance comparisons with standard LoRA

## Comparison with LoRA

| | Standard LoRA | SingLoRA |
|---|---|---|
| Decomposition | `W = W_0 + B @ A` (two matrices) | `W = W_0 + α/r * A @ A^T` (one matrix) |
| Trainable parameters | `2 * d * r` | `d * r` |
| Symmetry | none | A @ A^T is symmetric by construction |
| PEFT compatibility | native | drop-in via `setup_singlora()` |
| Typical use | adapters for any projection | parameter-constrained fine-tuning runs |

The practical effect: for the same rank, half the adapter parameters live in
the optimizer state, which matters most when you fine-tune large models on a
single GPU or when you stack adapters on many target modules.

## Requirements

- Python 3.9+ (CI runs 3.9 and 3.12)
- PyTorch 1.13+ (CUDA or CPU)
- `peft` 0.7+ and `transformers` 4.34+ (or newer)
- No flash-attn / triton requirements; the layer is pure PyTorch

## Development

```bash
git clone https://github.com/revaldianggara/PEFT-SingLoRA
cd PEFT-SingLoRA
pip install -e ".[dev]"
pytest tests/ -v
```

The suite covers registration with PEFT, forward/backward shape checks for
square and non-square projections, and the global-step update contract. Lint
with `flake8 src tests --max-line-length=100`.

## Roadmap

- [x] Single-matrix layer with PEFT registration
- [x] Non-square projection support (MLP up/down layers)
- [x] External global-step tracking (gradient accumulation friendly)
- [ ] Quantized base-model compatibility (int8 / 4-bit)
- [ ] Adapter merging / unmerging parity with LoRA

Contributions and issue reports against any roadmap item are welcome.

## Citation

If you use SingLoRA in your research, please cite:

```bibtex
@misc{bensaïd2025singloralowrankadaptation,
      title={SingLoRA: Low Rank Adaptation Using a Single Matrix}, 
      author={David Bensaïd and Noam Rotstein and Roy Velich and Daniel Bensaïd and Ron Kimmel},
      year={2025},
      eprint={2507.05566},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2507.05566}, 
}
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the BSD 2-Clause License - see the [LICENSE](LICENSE) file for details.
