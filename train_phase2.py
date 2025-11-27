"""
Phase 2: Style Transfer with Noise Injection

This phase applies conversational style using raw chat logs.
NEFTune noise injection improves generalization and prevents memorization.
"""
import argparse
from transformers import TrainingArguments
from trl import SFTTrainer

from config import PathConfig, ModelConfig, TrainingConfig
from data_utils import load_jsonl_dataset
from model_utils import (
    load_checkpoint,
    apply_lora,
    save_model,
    get_training_dtype
)


def train_phase2(
    path_config: PathConfig,
    model_config: ModelConfig,
    train_config: TrainingConfig,
    from_checkpoint: bool = True
) -> None:
    """
    Executes Phase 2 training pipeline.
    
    Args:
        from_checkpoint: If True, loads Phase 1 checkpoint. 
                        If False, starts from base model.
    """
    print("\n" + "=" * 50)
    print("[Phase 2] Style Transfer with Noise Injection")
    print("=" * 50)
    
    # Load model
    if from_checkpoint:
        model, tokenizer = load_checkpoint(
            path_config.output_phase1,
            model_config,
            train_config.seed
        )
    else:
        from model_utils import load_base_model
        model, tokenizer = load_base_model(model_config, train_config.seed)
    
    model = apply_lora(model, model_config)
    
    # Load dataset (using raw format parser)
    dataset = load_jsonl_dataset(
        path_config.phase2_data,
        tokenizer,
        use_raw_parser=True,
        min_length=10
    )
    
    if dataset is None:
        print("[ERROR] Failed to load Phase 2 dataset")
        return
    
    # Training arguments
    dtype_flags = get_training_dtype()
    
    training_args = TrainingArguments(
        output_dir=path_config.output_phase2,
        per_device_train_batch_size=train_config.phase2_batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=train_config.phase2_epochs,
        learning_rate=train_config.phase2_lr,
        weight_decay=train_config.phase2_weight_decay,
        logging_steps=10,
        save_steps=500,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        seed=train_config.seed,
        report_to="none",
        **dtype_flags,
    )
    
    # Initialize trainer with NEFTune
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=model_config.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
        neftune_noise_alpha=train_config.phase2_neftune_alpha,
    )
    
    # Train
    print("[INFO] Starting training with NEFTune noise injection...")
    print(f"  - NEFTune alpha: {train_config.phase2_neftune_alpha}")
    trainer.train()
    
    # Save
    save_model(model, tokenizer, path_config.output_phase2)
    print("[SUCCESS] Phase 2 completed")


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Style Transfer")
    parser.add_argument("--data", type=str, help="Path to raw chat data (JSONL)")
    parser.add_argument("--checkpoint", type=str, help="Path to Phase 1 checkpoint")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--neftune-alpha", type=float, default=5.0)
    parser.add_argument("--from-scratch", action="store_true", 
                       help="Start from base model instead of checkpoint")
    args = parser.parse_args()
    
    # Initialize configs
    path_config = PathConfig()
    model_config = ModelConfig()
    train_config = TrainingConfig()
    
    # Override with CLI arguments
    if args.data:
        path_config.phase2_data = args.data
    if args.checkpoint:
        path_config.output_phase1 = args.checkpoint
    if args.output:
        path_config.output_phase2 = args.output
    if args.epochs:
        train_config.phase2_epochs = args.epochs
    if args.lr:
        train_config.phase2_lr = args.lr
    if args.neftune_alpha:
        train_config.phase2_neftune_alpha = args.neftune_alpha
    
    train_phase2(
        path_config, 
        model_config, 
        train_config,
        from_checkpoint=not args.from_scratch
    )


if __name__ == "__main__":
    main()
