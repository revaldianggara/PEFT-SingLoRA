"""
Basic usage example for PEFT-SingLoRA.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from peft_singlora import setup_singlora, update_singlora_global_step


def main():
