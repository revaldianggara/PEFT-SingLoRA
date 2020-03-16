# PEFT-SingLoRA

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
