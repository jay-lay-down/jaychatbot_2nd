"""
Phase 1: Foundation Fine-tuning with Structured Data

This phase establishes the base persona using high-quality 
structured conversation data (Excel format).
"""
import argparse
from transformers import TrainingArguments
from trl import SFTTrainer

from config import PathConfig, ModelConfig, TrainingConfig
from data_utils import load_excel_dataset
from model_utils import (
    load_base_model, 
    apply_lora, 
    save_model,
    get_training_dtype
)


def train_phase1(
    path_config: PathConfig,
    model_config: ModelConfig,
    train_config: TrainingConfig
) -> None:
    """
    Executes Phase 1 training pipeline.
    """
    print("\n" + "=" * 50)
    print("[Phase 1] Foundation Fine-tuning")
    print("=" * 50)
    
    # Load model
    model, tokenizer = load_base_model(model_config, train_config.seed)
    model = apply_lora(model, model_config)
    
    # Load dataset
    dataset = load_excel_dataset(
        path_config.phase1_data, 
        tokenizer
    )
    
    if dataset is None:
        print("[ERROR] Failed to load Phase 1 dataset")
        return
    
    # Training arguments
    dtype_flags = get_training_dtype()
    
    training_args = TrainingArguments(
        output_dir=path_config.output_phase1,
        per_device_train_batch_size=train_config.phase1_batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=train_config.phase1_epochs,
        learning_rate=train_config.phase1_lr,
        weight_decay=train_config.phase1_weight_decay,
        logging_steps=10,
        save_steps=500,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        seed=train_config.seed,
        report_to="none",
        **dtype_flags,
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=model_config.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )
    
    # Train
    print("[INFO] Starting training...")
    trainer.train()
    
    # Save
    save_model(model, tokenizer, path_config.output_phase1)
    print("[SUCCESS] Phase 1 completed")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Foundation Fine-tuning")
    parser.add_argument("--data", type=str, help="Path to structured data (Excel)")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    
    # Initialize configs
    path_config = PathConfig()
    model_config = ModelConfig()
    train_config = TrainingConfig()
    
    # Override with CLI arguments
    if args.data:
        path_config.phase1_data = args.data
    if args.output:
        path_config.output_phase1 = args.output
    if args.epochs:
        train_config.phase1_epochs = args.epochs
    if args.lr:
        train_config.phase1_lr = args.lr
    if args.batch_size:
        train_config.phase1_batch_size = args.batch_size
    
    train_phase1(path_config, model_config, train_config)


if __name__ == "__main__":
    main()
