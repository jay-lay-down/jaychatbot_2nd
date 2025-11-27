"""
Data loading and preprocessing utilities
"""
import os
import pandas as pd
from typing import Optional, Callable
from datasets import Dataset, load_dataset, concatenate_datasets

from config import SYSTEM_PROMPT


def get_formatter(tokenizer) -> Callable:
    """
    Returns a formatting function that applies chat template with system prompt.
    """
    def format_chat(row: dict) -> dict:
        user_msg = row.get('Friend') or row.get('User') or row.get('instruction') or ""
        assist_msg = row.get('Me') or row.get('Assistant') or row.get('output') or ""
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(user_msg)},
            {"role": "assistant", "content": str(assist_msg)},
        ]
        
        formatted = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        return {"text": formatted}
    
    return format_chat


def parse_raw_chat_format(tokenizer) -> Callable:
    """
    Returns a parser for raw chat format (e.g., '### User: ... ### Assistant: ...')
    """
    def parse_and_format(row: dict) -> dict:
        raw_text = row.get('text', '')
        
        try:
            parts = raw_text.split("### Assistant:\n")
            if len(parts) < 2:
                return {"text": ""}
            
            user_msg = parts[0].replace("### User:\n", "").strip()
            assist_msg = parts[1].strip()
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assist_msg},
            ]
            
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            return {"text": formatted}
        
        except Exception:
            return {"text": ""}
    
    return parse_and_format


def load_excel_dataset(path: str, tokenizer, num_proc: int = 2) -> Optional[Dataset]:
    """
    Loads structured conversation data from Excel file.
    """
    if not os.path.exists(path):
        print(f"[WARNING] File not found: {path}")
        return None
    
    print(f"[INFO] Loading Excel dataset: {path}")
    df = pd.read_excel(path)
    ds = Dataset.from_pandas(df)
    
    formatter = get_formatter(tokenizer)
    ds = ds.map(formatter, num_proc=num_proc)
    
    print(f"[INFO] Loaded {len(ds)} samples")
    return ds


def load_jsonl_dataset(
    path: str, 
    tokenizer, 
    use_raw_parser: bool = False,
    min_length: int = 10,
    num_proc: int = 2
) -> Optional[Dataset]:
    """
    Loads conversation data from JSONL file.
    
    Args:
        path: Path to JSONL file
        tokenizer: Tokenizer for chat template
        use_raw_parser: If True, uses raw chat format parser
        min_length: Minimum text length filter
        num_proc: Number of processes for mapping
    """
    if not os.path.exists(path):
        print(f"[WARNING] File not found: {path}")
        return None
    
    print(f"[INFO] Loading JSONL dataset: {path}")
    ds = load_dataset("json", data_files=path, split="train")
    
    if use_raw_parser:
        parser = parse_raw_chat_format(tokenizer)
    else:
        parser = get_formatter(tokenizer)
    
    ds = ds.map(parser, num_proc=num_proc)
    
    # Filter short/empty sequences
    ds = ds.filter(lambda x: len(x["text"]) > min_length)
    
    print(f"[INFO] Loaded {len(ds)} samples after filtering")
    return ds


def load_and_augment_dataset(
    path: str,
    tokenizer,
    augment_factor: int = 1,
    num_proc: int = 2
) -> Optional[Dataset]:
    """
    Loads dataset and applies augmentation by replication.
    Used for reinforcing specific behavioral patterns.
    """
    ds = load_jsonl_dataset(path, tokenizer, num_proc=num_proc)
    
    if ds is None:
        return None
    
    if augment_factor > 1:
        print(f"[INFO] Augmenting dataset x{augment_factor}")
        ds = concatenate_datasets([ds] * augment_factor)
    
    return ds


def merge_datasets(*datasets, seed: int = 42) -> Dataset:
    """
    Merges multiple datasets and shuffles.
    """
    valid_datasets = [ds for ds in datasets if ds is not None]
    
    if not valid_datasets:
        raise ValueError("No valid datasets provided")
    
    combined = concatenate_datasets(valid_datasets)
    combined = combined.shuffle(seed=seed)
    
    print(f"[INFO] Combined dataset size: {len(combined)}")
    return combined
