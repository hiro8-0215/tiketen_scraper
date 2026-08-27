"""Train leakage-safe Qwen OOF predictions on the Model13-clean population."""
from __future__ import annotations

import argparse
import gc
import os
import time

# Resource policy must be configured before importing torch/CUDA.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,garbage_collection_threshold:0.80",
)
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
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainingArguments,
)

from config import (
    ARTIFACT_DIR,
    N_FOLDS,
    QWEN_BATCH_SIZE,
    QWEN_CPU_THREADS,
    QWEN_DATALOADER_WORKERS,
    QWEN_EPOCHS,
    QWEN_EVAL_BATCH_SIZE,
    QWEN_GPU_INDEX,
    QWEN_GPU_MEMORY_FRACTION,
    QWEN_GRAD_ACCUM,
    QWEN_LR,
    QWEN_MAX_LENGTH,
    QWEN_MODEL,
    QWEN_OOF_SCHEMA_VERSION,
    SEED,
    TARGET,
)
from data_loader import prepare_dataset
from qwen_prompt import build_qwen_prompt, qwen_dataset_fingerprint
from qwen_trainer import OrderedEvalTrainer, qwen_regression_metrics
from qwen_validation import attach_qwen_diagnostics

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.set_num_threads(QWEN_CPU_THREADS)
torch.set_num_interop_threads(2)


def configure_resources():
    if not torch.cuda.is_available():
        raise RuntimeError("Model15 Qwen OOF requires an NVIDIA CUDA GPU")
    torch.cuda.set_device(QWEN_GPU_INDEX)
    torch.cuda.set_per_process_memory_fraction(QWEN_GPU_MEMORY_FRACTION, QWEN_GPU_INDEX)
    properties = torch.cuda.get_device_properties(QWEN_GPU_INDEX)
    print(
        f"[resource] GPU={properties.name}, dedicated={properties.total_memory / 2**30:.1f}GiB, "
        f"allocator_limit={QWEN_GPU_MEMORY_FRACTION:.0%}, CPU_threads={QWEN_CPU_THREADS}"
    )


def assert_gpu_only(model):
    device_map = getattr(model, "hf_device_map", {}) or {}
    bad_map = {
        name: device for name, device in device_map.items()
        if str(device).lower() in {"cpu", "disk"}
    }
    non_cuda = [
        name for name, parameter in model.named_parameters()
        if parameter.device.type != "cuda"
    ]
    if bad_map or non_cuda:
        raise RuntimeError(
            "CPU/disk offload detected; shared-memory training is disabled. "
            f"device_map={bad_map}, non_cuda_parameters={non_cuda[:5]}"
        )


def dataset_for(df):
    return Dataset.from_dict({
        "text": [build_qwen_prompt(row) for _, row in df.iterrows()],
        "labels": np.log1p(df[TARGET].to_numpy(float)).astype(np.float32),
    })


def load_base():
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForSequenceClassification.from_pretrained(
        QWEN_MODEL,
        num_labels=1,
        problem_type="regression",
        quantization_config=quantization,
        device_map={"": QWEN_GPU_INDEX},
        attn_implementation="sdpa",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = get_peft_model(base, LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    ))
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    assert_gpu_only(model)
    return tokenizer, model


def tokenize(dataset, tokenizer):
    def encode(batch):
        encoded = tokenizer(batch["text"], truncation=True, max_length=QWEN_MAX_LENGTH)
        encoded["length"] = [len(ids) for ids in encoded["input_ids"]]
        return encoded

    return dataset.map(encode, batched=True, batch_size=256).remove_columns(["text"])


def train_one(train_df, validation_df, output_dir, train_batch, grad_accum, eval_batch):
    tokenizer, model = load_base()
    train_dataset = tokenize(dataset_for(train_df), tokenizer)
    validation_dataset = tokenize(dataset_for(validation_df), tokenizer)
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        seed=SEED,
        learning_rate=QWEN_LR,
        per_device_train_batch_size=train_batch,
        per_device_eval_batch_size=eval_batch,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=QWEN_EPOCHS,
        warmup_ratio=0.05,
        weight_decay=0.05,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=False,
        fp16=True,
        tf32=True,
        optim="adamw_torch_fused",
        train_sampling_strategy="group_by_length",
        length_column_name="length",
        dataloader_num_workers=QWEN_DATALOADER_WORKERS,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_mae_yen",
        greater_is_better=False,
        prediction_loss_only=False,
        logging_steps=20,
        report_to="none",
    )
    trainer = OrderedEvalTrainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
        compute_metrics=qwen_regression_metrics,
    )
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))
    started = time.time()
    trainer.train()
    predictions = trainer.predict(validation_dataset).predictions.squeeze().astype(float)
    trainer.save_model(str(output_dir / "best_adapter"))
    (output_dir / "elapsed_hours.txt").write_text(
        f"{(time.time() - started) / 3600:.4f}\n", encoding="utf-8"
    )
    del trainer, model, tokenizer, train_dataset, validation_dataset
    gc.collect()
    torch.cuda.empty_cache()
    return np.atleast_1d(predictions)


def fresh_result(manifest, fingerprint):
    result = manifest[["ticket_id", "fold"]].copy()
    result["qwen_dataset_fingerprint"] = fingerprint
    result["qwen_oof_schema_version"] = QWEN_OOF_SCHEMA_VERSION
    result["qwen_training_source"] = "fresh_fp16_ordered_training"
    result["qwen_pred_log"] = np.nan
    return result


def save_result(result, target):
    temporary = target.with_name("qwen_oof.model15.tmp.csv")
    result.to_csv(temporary, index=False)
    # Atomic replacement prevents an old hard-link from modifying Model14.
    os.replace(temporary, target)


def main():
    configure_resources()
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(N_FOLDS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    df = prepare_dataset()
    manifest = pd.read_csv(ARTIFACT_DIR / "folds.csv")
    if (
        not {"ticket_id", "duplicate_group", "fold"}.issubset(manifest.columns)
        or manifest.ticket_id.tolist() != df.ticket_id.tolist()
        or manifest.duplicate_group.tolist() != df.duplicate_group.tolist()
    ):
        raise ValueError("folds.csv does not match clean Model15 data; run make_folds.py")
    fingerprint = qwen_dataset_fingerprint(df, manifest)

    prediction_path = ARTIFACT_DIR / "qwen_oof.csv"
    result = fresh_result(manifest, fingerprint)
    if prediction_path.exists():
        previous = pd.read_csv(prediction_path)
        expected = manifest[["ticket_id", "fold"]].reset_index(drop=True)
        actual = previous[["ticket_id", "fold"]].reset_index(drop=True) if {
            "ticket_id", "fold", "qwen_dataset_fingerprint",
            "qwen_oof_schema_version", "qwen_pred_log"
        }.issubset(previous.columns) else None
        fingerprint_matches = (
            actual is not None
            and previous["qwen_dataset_fingerprint"].eq(fingerprint).all()
            and previous["qwen_oof_schema_version"].eq(QWEN_OOF_SCHEMA_VERSION).all()
        )
        if actual is not None and actual.equals(expected) and fingerprint_matches:
            result = previous
        else:
            print("[resume] Qwen inputs/labels/folds changed; starting a clean OOF manifest")

    selected_folds = [args.fold] if args.fold is not None else list(range(N_FOLDS))
    for fold in selected_folds:
        validation_mask = manifest.fold.eq(fold).to_numpy()
        if not args.force and result.loc[validation_mask, "qwen_pred_log"].notna().all():
            print(f"fold {fold}: completed; skipping")
            continue
        fold_dir = ARTIFACT_DIR / "qwen_folds" / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        try:
            fold_predictions = train_one(
                df.loc[~validation_mask],
                df.loc[validation_mask],
                fold_dir,
                QWEN_BATCH_SIZE,
                QWEN_GRAD_ACCUM,
                QWEN_EVAL_BATCH_SIZE,
            )
        except RuntimeError as error:
            if "out of memory" not in str(error).lower() or QWEN_BATCH_SIZE <= 2:
                raise
            print("[resource] VRAM limit reached; retrying batch=2/accum=16")
            gc.collect()
            torch.cuda.empty_cache()
            fold_predictions = train_one(
                df.loc[~validation_mask], df.loc[validation_mask], fold_dir, 2, 16, 4
            )
        result.loc[validation_mask, "qwen_pred_log"] = fold_predictions
        save_result(result, prediction_path)

    print(result.groupby("fold").qwen_pred_log.count())
    if result["qwen_pred_log"].notna().all():
        diagnostics = attach_qwen_diagnostics(result, df, manifest)
        save_result(result, prediction_path)
        print(f"ordered Qwen OOF diagnostics: {diagnostics}")


if __name__ == "__main__":
    main()
