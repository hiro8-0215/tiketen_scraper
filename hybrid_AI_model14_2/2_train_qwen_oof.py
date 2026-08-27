"""Optimized 5-fold Qwen 7B QLoRA without reducing data, folds, epochs or context."""
from __future__ import annotations
import argparse
import gc
import os
import time

# These must be set before importing torch/CUDA. They prevent allocator
# fragmentation and CPU thread oversubscription on Windows.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.80")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig,
    DataCollatorWithPadding, EarlyStoppingCallback, Trainer, TrainingArguments,
)
from config import (
    ARTIFACT_DIR, N_FOLDS, QWEN_BATCH_SIZE, QWEN_DATALOADER_WORKERS,
    QWEN_CPU_THREADS, QWEN_GPU_INDEX, QWEN_GPU_MEMORY_FRACTION,
    QWEN_EPOCHS, QWEN_EVAL_BATCH_SIZE, QWEN_GRAD_ACCUM, QWEN_LR,
    QWEN_MAX_LENGTH, QWEN_MODEL, SEED, TARGET,
)
from data_loader import prepare_dataset

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.set_num_threads(QWEN_CPU_THREADS)
torch.set_num_interop_threads(2)


def configure_process_resources():
    if not torch.cuda.is_available():
        raise RuntimeError("Model 14-2 requires CUDA; CPU fallback is intentionally disabled")
    torch.cuda.set_device(QWEN_GPU_INDEX)
    # Keep dedicated VRAM headroom. PyTorch will raise CUDA OOM instead of
    # silently growing into slow WDDM shared system memory.
    torch.cuda.set_per_process_memory_fraction(QWEN_GPU_MEMORY_FRACTION, QWEN_GPU_INDEX)
    try:
        import psutil
        process = psutil.Process()
        if os.name == "nt":
            process.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
    except (ImportError, OSError, PermissionError):
        pass
    props = torch.cuda.get_device_properties(QWEN_GPU_INDEX)
    print(
        f"[resource] GPU={props.name}, dedicated={props.total_memory / 2**30:.1f}GiB, "
        f"allocator_limit={QWEN_GPU_MEMORY_FRACTION:.0%}, CPU_threads={torch.get_num_threads()}, "
        f"dataloader_workers={QWEN_DATALOADER_WORKERS}"
    )


def assert_gpu_only(model):
    device_map = getattr(model, "hf_device_map", {}) or {}
    bad_map = {name: device for name, device in device_map.items()
               if str(device).lower() in {"cpu", "disk"}}
    cpu_parameters = [name for name, parameter in model.named_parameters()
                      if parameter.device.type != "cuda"]
    if bad_map or cpu_parameters:
        raise RuntimeError(
            "CPU/disk offload detected; refusing slow shared-memory training. "
            f"device_map={bad_map}, non_cuda_parameters={cpu_parameters[:5]}"
        )
    print(f"[resource] GPU-only placement verified: {device_map}")


def prompt(row):
    def value(name, default="不明"):
        v = row.get(name)
        return default if pd.isna(v) or str(v) == "" else str(v)
    return (
        "次の売却済みチケットの成立価格を推定してください。\n"
        f"アーティスト:{value('group_slug')}\n公演:{value('event_id')}\n"
        f"会場:{value('venue')}\n公演日:{value('perf_date')}\n定価:{value('base_price')}\n"
        f"券種:{value('ticket_type')}\n名義:{value('name_type')}\n枚数:{value('quantity')}\n"
        f"過去成立中央値:{value('event_prior_sold_median')}\n"
        f"出品時点までの出品数:{value('event_listings_seen_before')}\n"
        f"説明:{value('raw_description', '')}\nタグ:{value('ticket_tags', '')}"
    )


def dataset_for(df):
    return Dataset.from_dict({
        "text": [prompt(row) for _, row in df.iterrows()],
        "labels": np.log1p(df[TARGET].to_numpy(float)).astype(np.float32),
    })


def load_base():
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForSequenceClassification.from_pretrained(
        QWEN_MODEL, num_labels=1, problem_type="regression", quantization_config=quant,
        device_map={"": QWEN_GPU_INDEX}, attn_implementation="sdpa", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = get_peft_model(base, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none",
    ))
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    assert_gpu_only(model)
    return tokenizer, model


def tokenize(ds, tokenizer):
    def encode(batch):
        encoded = tokenizer(batch["text"], truncation=True, max_length=QWEN_MAX_LENGTH)
        encoded["length"] = [len(ids) for ids in encoded["input_ids"]]
        return encoded
    # Tokenization is CPU work and is completed before the GPU training loop.
    return ds.map(encode, batched=True, batch_size=256).remove_columns(["text"])


def train_one(train_df, val_df, output_dir, train_batch, grad_accum, eval_batch):
    tokenizer, model = load_base()
    train_ds = tokenize(dataset_for(train_df), tokenizer)
    val_ds = tokenize(dataset_for(val_df), tokenizer)
    args = TrainingArguments(
        output_dir=str(output_dir), seed=SEED, learning_rate=QWEN_LR,
        per_device_train_batch_size=train_batch,
        per_device_eval_batch_size=eval_batch,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=QWEN_EPOCHS, warmup_ratio=0.05, weight_decay=0.05,
        lr_scheduler_type="cosine", gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True, fp16=False, tf32=True, optim="adamw_torch_fused",
        train_sampling_strategy="group_by_length", length_column_name="length",
        dataloader_num_workers=QWEN_DATALOADER_WORKERS,
        dataloader_pin_memory=True, dataloader_persistent_workers=True,
        dataloader_prefetch_factor=2,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, prediction_loss_only=True,
        logging_steps=20, report_to="none",
    )
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      processing_class=tokenizer, data_collator=collator)
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))
    started = time.time()
    trainer.train()
    # Training evaluation collects loss only; final OOF inference needs logits.
    trainer.args.prediction_loss_only = False
    pred = trainer.predict(val_ds).predictions.squeeze().astype(float)
    trainer.save_model(str(output_dir / "best_adapter"))
    elapsed = (time.time() - started) / 3600
    (output_dir / "elapsed_hours.txt").write_text(f"{elapsed:.4f}\n", encoding="utf-8")
    del trainer, model, tokenizer, train_ds, val_ds
    gc.collect()
    torch.cuda.empty_cache()
    return np.atleast_1d(pred)


def main():
    configure_process_resources()
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, help="Run one fold; omit for all folds")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    df = prepare_dataset()
    manifest = pd.read_csv(ARTIFACT_DIR / "folds.csv")
    if not manifest.ticket_id.equals(df.ticket_id):
        raise ValueError("folds.csv mismatch; run bootstrap_artifacts.py")
    pred_path = ARTIFACT_DIR / "qwen_oof.csv"
    result = pd.read_csv(pred_path) if pred_path.exists() else manifest.assign(qwen_pred_log=np.nan)
    folds = [args.fold] if args.fold is not None else list(range(N_FOLDS))
    for fold in folds:
        val_mask = manifest.fold.eq(fold).to_numpy()
        if not args.force and result.loc[val_mask, "qwen_pred_log"].notna().all():
            print(f"fold {fold}: 完了済みのためスキップ")
            continue
        fold_dir = ARTIFACT_DIR / "qwen_folds" / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        try:
            fold_predictions = train_one(
                df.loc[~val_mask], df.loc[val_mask], fold_dir,
                QWEN_BATCH_SIZE, QWEN_GRAD_ACCUM, QWEN_EVAL_BATCH_SIZE,
            )
        except RuntimeError as error:
            if "out of memory" not in str(error).lower() or QWEN_BATCH_SIZE <= 2:
                raise
            # Keep the same effective batch (2 * 16 = 4 * 8 = 32). This is a
            # resource-safe fallback, not a reduction in training volume.
            print("[resource] dedicated VRAM limit reached; retrying batch=2/accum=16")
            gc.collect()
            torch.cuda.empty_cache()
            fold_predictions = train_one(df.loc[~val_mask], df.loc[val_mask], fold_dir, 2, 16, 4)
        result.loc[val_mask, "qwen_pred_log"] = fold_predictions
        result.to_csv(pred_path, index=False)
    print(result.groupby("fold").qwen_pred_log.count())


if __name__ == "__main__":
    main()
