"""
Basic usage example for PEFT-SingLoRA.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from peft_singlora import setup_singlora, update_singlora_global_step


def main():
    setup_singlora()
    model_name = "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = LoraConfig(
        r=8,
        lora_alpha=32,
