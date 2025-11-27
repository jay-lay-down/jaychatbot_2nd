"""
Phase 3: Behavioral Alignment with Data Augmentation

This phase reinforces specific behavioral patterns using:
- High augmentation (x10) for correction data
- Low augmentation (x2) for style maintenance
"""
import argparse
from transformers import TrainingArguments
from trl import SFTTrainer

from config import PathConfig, ModelConfig, TrainingConfig
from data_utils import load_and_augment_dataset, merge_datasets
from model_utils import (
    load_checkpoint,
    apply_lora,
    save_model,
    get_training_dtype
)


def train_phase3(
    path_config: PathConfig,
    model_config: ModelConfig,
    train_config: TrainingConfig
) -> None:
    """
    Executes Phase 3 training pipeline.
    """
    print("\n" + "=" * 50)
    print("[Phase 3] Behavioral Alignment")
    print("=" * 50)
    
    # Load model from Phase 2 checkpoint
    model, tokenizer = load_checkpoint(
        path_config.output_phase2,
        model_config,
        train_config.seed
    )
    model = apply_lora(model, model_config)
    
    # Load and augment datasets
    print("[INFO] Loading correction data with high augmentation...")
    ds_correction = load_and_augment_dataset(
        path_config.phase3_correction_data,
        tokenizer,
        augment_factor=train_config.phase3_correction_augment
    )
    
    print("[INFO] Loading style data with low augmentation...")
    ds_style = load_and_augment_dataset(
        path_config.phase3_style_data,
        tokenizer,
        augment_factor=train_config.phase3_style_augment
    )
    
    # Merge datasets
    try:
        dataset = merge_datasets(ds_correction, ds_style, seed=train_config.seed)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return
    
    # Training arguments
    dtype_flags = get_training_dtype()
    
    training_args = TrainingArguments(
        output_dir=path_config.output_phase3,
        per_device_train_batch_size=train_config.phase3_batch_size,
        gradient_accumulation_steps=train_config.phase3_grad_accum,
        num_train_epochs=train_config.phase3_epochs,
        learning_rate=train_config.phase3_lr,
        weight_decay=train_config.phase3_weight_decay,
        logging_steps=10,
        save_steps=200,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        seed=train_config.seed,
        report_to="none",
        **dtype_flags,
    )
    
    # Initialize trainer (NEFTune disabled for sharp alignment)
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
    print("[INFO] Starting behavioral alignment training...")
    trainer.train()
    
    # Save
    save_model(model, tokenizer, path_config.output_phase3)
    print("[SUCCESS] Phase 3 completed")


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Behavioral Alignment")
    parser.add_argument("--correction-data", type=str, help="Path to correction data")
    parser.add_argument("--style-data", type=str, help="Path to style data")
    parser.add_argument("--checkpoint", type=str, help="Path to Phase 2 checkpoint")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--correction-augment", type=int, default=10)
    parser.add_argument("--style-augment", type=int, default=2)
    args = parser.parse_args()
    
    # Initialize configs
    path_config = PathConfig()
    model_config = ModelConfig()
    train_config = TrainingConfig()
    
    # Override with CLI arguments
    if args.correction_data:
        path_config.phase3_correction_data = args.correction_data
    if args.style_data:
        path_config.phase3_style_data = args.style_data
    if args.checkpoint:
        path_config.output_phase2 = args.checkpoint
    if args.output:
        path_config.output_phase3 = args.output
    if args.epochs:
        train_config.phase3_epochs = args.epochs
    if args.lr:
        train_config.phase3_lr = args.lr
    if args.correction_augment:
        train_config.phase3_correction_augment = args.correction_augment
    if args.style_augment:
        train_config.phase3_style_augment = args.style_augment
    
    train_phase3(path_config, model_config, train_config)


if __name__ == "__main__":
    main()
