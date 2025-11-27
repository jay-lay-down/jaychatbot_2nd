# jaychatbot_2nd
👿 AI assistant? No. I'm your childhood friend who just happens to live in GPU (in Korean)
huggingface: https://huggingface.co/spaces/Jay1121/immitate_chatbot


The Ultimate Sassy Korean Best Friend AI
Ddolbae is a custom fine-tuned LLM designed to simulate a realistic, cynical, and witty close friend. Unlike standard polite AI models, Ddolbae speaks in Banmal (casual Korean), uses slang, and isn't afraid to roast you—but deep down, he gives logically sound advice (sometimes).
Built on Qwen2.5-1.5B-Instruct using Unsloth, this model focuses on maximizing conversational realism while maintaining a distinct "Tsundere" personality.

✨ Key Persona Characteristics
No Honorifics (존댓말 금지) : Ddolbae strictly uses Banmal (informal speech). He treats you like a friend he's known for 20 years.
Sassy & Blunt (독설과 팩폭): He doesn't sugarcoat things. If you say something stupid, he will roast you. He uses Korean internet slang and mild profanity naturally.
Logical sarcasm: He's not just an angry bot. He counters user arguments with cynicism and logic.
Anti-Meta: He denies being an AI. He acts like a real person texting you on KakaoTalk.

🏗️ Training Architecture (The "3-Step" Pipeline)

## Overview
This project implements a multi-phase training approach to inject a distinct personality ("똘배" - a cynical, informal Korean friend persona) into a small language model (Qwen2.5-1.5B).

### Training Phases

| Phase | Purpose | Data Type | Key Technique |
|-------|---------|-----------|---------------|
| Phase 1 | Foundation | Structured Excel | Standard SFT |
| Phase 2 | Style Transfer | Raw Chat Logs | NEFTune Noise Injection |
| Phase 3 | Behavioral Alignment | Mixed + Augmented | Data Augmentation |

## Project Structure
```
├── config.py           # All configurations and hyperparameters
├── data_utils.py       # Data loading and preprocessing
├── model_utils.py      # Model loading and LoRA setup
├── train_phase1.py     # Phase 1: Foundation training
├── train_phase2.py     # Phase 2: Style transfer
├── train_phase3.py     # Phase 3: Behavioral alignment
├── inference.py        # Interactive chat & GGUF export
├── requirements.txt    # Dependencies
└── README.md
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage

🛠️ Tech Stack
Base Model: Qwen/Qwen2.5-1.5B-Instruct
Library: Unsloth (for 2x faster training & 60% less memory usage)
Method: LoRA (Low-Rank Adaptation) / QLoRA (4bit)
Key Parameters: r=64, alpha=128 (High capacity for style transfer)

### Prepare Data

Organize your data in the following structure:
```
data/
├── structured_chat.xlsx    # Phase 1: Columns [Friend/User, Me/Assistant]
├── raw_chat.jsonl          # Phase 2: Raw conversation format
├── correction_data.jsonl   # Phase 3: High-quality corrections
└── style_data.jsonl        # Phase 3: Style maintenance
```

### Run Training
```bash
# Phase 1: Foundation
python train_phase1.py --data ./data/structured_chat.xlsx --epochs 3

# Phase 2: Style Transfer
python train_phase2.py --data ./data/raw_chat.jsonl --neftune-alpha 5.0

# Phase 3: Behavioral Alignment
python train_phase3.py \
    --correction-data ./data/correction_data.jsonl \
    --style-data ./data/style_data.jsonl \
    --correction-augment 10 \
    --style-augment 2
```

### Inference & Export
```bash
# Interactive chat
python inference.py --checkpoint ./outputs/phase3_final

# Export to GGUF and upload to HuggingFace
python inference.py \
    --checkpoint ./outputs/phase3_final \
    --hf-repo username/model-name \
    --quantization q4_k_m
```

## Configuration

All hyperparameters can be modified in `config.py` or via CLI arguments.

### Key Hyperparameters

| Parameter | Phase 1 | Phase 2 | Phase 3 |
|-----------|---------|---------|---------|
| Learning Rate | 2e-4 | 1e-4 | 8e-5 |
| Weight Decay | 0.01 | 0.1 | 0.01 |
| Epochs | 3 | 3 | 3 |
| NEFTune Alpha | - | 5.0 | - |

### LoRA Configuration
- Rank (r): 64
- Alpha: 128
- Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

## Environment Variables
```bash
export PHASE1_DATA_PATH="./data/structured_chat.xlsx"
export PHASE2_DATA_PATH="./data/raw_chat.jsonl"
export PHASE3_CORRECTION_PATH="./data/correction_data.jsonl"
export PHASE3_STYLE_PATH="./data/style_data.jsonl"
```
⚠️ Disclaimer
Language Warning: This model is trained to generate profanity, slang, and aggressive text for entertainment purposes. It is not suitable for formal environments or children.
Hallucination: Like all LLMs, Ddolbae may generate incorrect information with high confidence. Don't take his financial or life advice too seriously.

👨‍💻 Author
Developed by Jihee Cho (Jay1121)
Powered by Unsloth AI

## License

Apache License

## Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning
- [Qwen](https://github.com/QwenLM/Qwen) for the base model
