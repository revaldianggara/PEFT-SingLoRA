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

