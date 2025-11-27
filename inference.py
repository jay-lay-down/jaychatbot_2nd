"""
Inference and GGUF Export Utilities

Interactive chat interface and model conversion for deployment.
"""
import argparse
import os
from typing import Optional

import torch
from transformers import TextStreamer
from huggingface_hub import HfApi, login

from config import (
    PathConfig, 
    ModelConfig, 
    TrainingConfig,
    InferenceConfig,
    HFConfig,
    SYSTEM_PROMPT
)
from model_utils import load_checkpoint, prepare_for_inference


def generate_response(
    model,
    tokenizer,
    user_input: str,
    inference_config: InferenceConfig,
    stream: bool = True
) -> str:
    """
    Generates a response from the model.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    generation_kwargs = {
        "input_ids": inputs,
        "max_new_tokens": inference_config.max_new_tokens,
        "use_cache": True,
        "temperature": inference_config.temperature,
        "top_p": inference_config.top_p,
        "repetition_penalty": inference_config.repetition_penalty,
    }
    
    if stream:
        streamer = TextStreamer(tokenizer, skip_prompt=True)
        generation_kwargs["streamer"] = streamer
    
    outputs = model.generate(**generation_kwargs)
    
    if not stream:
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract assistant response
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        return response
    
    return ""


def export_to_gguf(
    model,
    tokenizer,
    output_dir: str,
    hf_config: HFConfig
) -> None:
    """
    Converts model to GGUF format and optionally uploads to Hugging Face.
    """
    print(f"\n[INFO] Converting to GGUF ({hf_config.gguf_quantization})...")
    
    model.save_pretrained_gguf(
        output_dir,
        tokenizer,
        quantization_method=hf_config.gguf_quantization
    )
    
    if hf_config.repo_id:
        print(f"[INFO] Uploading to Hugging Face: {hf_config.repo_id}")
        
        try:
            login()
            api = HfApi()
            api.create_repo(
                repo_id=hf_config.repo_id,
                exist_ok=True,
                repo_type="model"
            )
            
            # Find GGUF file
            gguf_files = [f for f in os.listdir(output_dir) if f.endswith('.gguf')]
            
            for gguf_file in gguf_files:
                api.upload_file(
                    path_or_fileobj=os.path.join(output_dir, gguf_file),
                    path_in_repo=gguf_file,
                    repo_id=hf_config.repo_id,
                    repo_type="model",
                )
                print(f"[SUCCESS] Uploaded: {gguf_file}")
                
        except Exception as e:
            print(f"[ERROR] Upload failed: {e}")


def interactive_chat(
    model,
    tokenizer,
    inference_config: InferenceConfig,
    output_dir: str,
    hf_config: Optional[HFConfig] = None
) -> None:
    """
    Starts interactive chat session.
    
    Commands:
        - 'exit' or 'quit': End session
        - 'save': Export to GGUF and upload
    """
    print("\n" + "=" * 50)
    print("🧪 Interactive Chat Session")
    print("Commands: 'exit' to quit, 'save' to export GGUF")
    print("=" * 50 + "\n")
    
    prepare_for_inference(model)
    
    while True:
        try:
            user_input = input("User: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[INFO] Session ended")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ["exit", "quit"]:
            print("[INFO] Goodbye!")
            break
        
        if user_input.lower() == "save":
            if hf_config:
                export_to_gguf(model, tokenizer, output_dir, hf_config)
            else:
                print("[WARNING] No HF config provided. Skipping upload.")
                hf_config = HFConfig()
                export_to_gguf(model, tokenizer, output_dir, hf_config)
            continue
        
        print("Assistant: ", end="")
        generate_response(model, tokenizer, user_input, inference_config)
        print()


def main():
    parser = argparse.ArgumentParser(description="Inference & Export")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--hf-repo", type=str, default=None,
                       help="Hugging Face repo ID for upload")
    parser.add_argument("--quantization", type=str, default="q4_k_m",
                       help="GGUF quantization method")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--no-stream", action="store_true",
                       help="Disable streaming output")
    args = parser.parse_args()
    
    # Initialize configs
    model_config = ModelConfig()
    inference_config = InferenceConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_tokens
    )
    hf_config = HFConfig(
        repo_id=args.hf_repo,
        gguf_quantization=args.quantization
    ) if args.hf_repo else None
    
    # Load model
    model, tokenizer = load_checkpoint(
        args.checkpoint,
        model_config
    )
    
    # Start chat
    interactive_chat(
        model,
        tokenizer,
        inference_config,
        args.checkpoint,
        hf_config
    )


if __name__ == "__main__":
    main()
```

---

**requirements.txt**
```
unsloth @ git+https://github.com/unslothai/unsloth.git
trl>=0.7.0
peft>=0.6.0
accelerate>=0.24.0
bitsandbytes>=0.41.0
transformers>=4.36.0
datasets>=2.14.0
pandas>=2.0.0
openpyxl>=3.1.0
huggingface_hub>=0.19.0
torch>=2.0.0
xformers
