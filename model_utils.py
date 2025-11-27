"""
Model loading and configuration utilities
"""
import torch
from typing import Tuple, Any
from unsloth import FastLanguageModel
from transformers import set_seed

from config import ModelConfig, TrainingConfig


def load_base_model(
    config: ModelConfig,
    seed: int = 42
) -> Tuple[Any, Any]:
    """
    Loads base model with Unsloth optimizations.
    
    Returns:
        Tuple of (model, tokenizer)
    """
    set_seed(seed)
    
    print(f"[INFO] Loading base model: {config.base_model_id}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model_id,
        max_seq_length=config.max_seq_length,
        dtype=None,  # Auto-detection
        load_in_4bit=config.load_in_4bit,
    )
    
    return model, tokenizer


def load_checkpoint(
    checkpoint_path: str,
    config: ModelConfig,
    seed: int = 42
) -> Tuple[Any, Any]:
    """
    Loads model from a previous checkpoint.
    
    Returns:
        Tuple of (model, tokenizer)
    """
    set_seed(seed)
    
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=config.load_in_4bit,
    )
    
    return model, tokenizer


def apply_lora(model: Any, config: ModelConfig) -> Any:
    """
    Applies LoRA configuration to the model.
    
    Uses high rank (r=64) and alpha (128) for maximum 
    style transfer and persona injection capabilities.
    """
    print("[INFO] Applying LoRA configuration")
    print(f"  - r: {config.lora_r}, alpha: {config.lora_alpha}")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=config.target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    return model


def save_model(model: Any, tokenizer: Any, output_dir: str) -> None:
    """
    Saves model and tokenizer to specified directory.
    """
    print(f"[INFO] Saving model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def get_training_dtype() -> dict:
    """
    Returns appropriate dtype flags based on hardware support.
    """
    bf16_supported = torch.cuda.is_bf16_supported()
    
    return {
        "fp16": not bf16_supported,
        "bf16": bf16_supported,
    }


def prepare_for_inference(model: Any) -> Any:
    """
    Prepares model for inference mode.
    """
    FastLanguageModel.for_inference(model)
    return model
