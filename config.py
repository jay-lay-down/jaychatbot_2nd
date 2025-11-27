"""
Configuration settings for Korean Persona Chatbot Fine-tuning Pipeline
"""
import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class PathConfig:
    """Dataset and output path configurations"""
    # Dataset Paths (Set via environment variables or modify defaults)
    phase1_data: str = os.environ.get("PHASE1_DATA_PATH", "./data/structured_chat.xlsx")
    phase2_data: str = os.environ.get("PHASE2_DATA_PATH", "./data/raw_chat.jsonl")
    phase3_correction_data: str = os.environ.get("PHASE3_CORRECTION_PATH", "./data/correction_data.jsonl")
    phase3_style_data: str = os.environ.get("PHASE3_STYLE_PATH", "./data/style_data.jsonl")
    
    # Output Directories
    output_phase1: str = "./outputs/phase1_foundation"
    output_phase2: str = "./outputs/phase2_style_transfer"
    output_phase3: str = "./outputs/phase3_final"

@dataclass
class ModelConfig:
    """Model and LoRA configurations"""
    base_model_id: str = "unsloth/Qwen2.5-1.5B-Instruct"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    
    # LoRA Parameters
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

@dataclass
class TrainingConfig:
    """Training hyperparameters for each phase"""
    seed: int = 42
    
    # Phase 1: Foundation
    phase1_epochs: int = 3
    phase1_lr: float = 2e-4
    phase1_batch_size: int = 16
    phase1_weight_decay: float = 0.01
    
    # Phase 2: Style Transfer
    phase2_epochs: int = 3
    phase2_lr: float = 1e-4
    phase2_batch_size: int = 16
    phase2_weight_decay: float = 0.1
    phase2_neftune_alpha: float = 5.0
    
    # Phase 3: Behavioral Alignment
    phase3_epochs: int = 3
    phase3_lr: float = 8e-5
    phase3_batch_size: int = 8
    phase3_grad_accum: int = 2
    phase3_weight_decay: float = 0.01
    
    # Data Augmentation
    phase3_correction_augment: int = 10
    phase3_style_augment: int = 2

@dataclass
class InferenceConfig:
    """Inference generation parameters"""
    max_new_tokens: int = 256
    temperature: float = 0.9
    top_p: float = 0.9
    repetition_penalty: float = 1.2

# System Prompt Definition
SYSTEM_PROMPT = (
    "너는 사용자의 '불알친구'인 '똘배'다. "
    "메타 발언(AI, 챗봇 등)은 절대 하지 말고, "
    "20대 남자 사람 친구처럼 반말과 비속어를 섞어 자연스럽게 대화해라."
)

# Hugging Face Upload Config (Optional)
@dataclass
class HFConfig:
    repo_id: Optional[str] = None
    gguf_quantization: str = "q4_k_m"
