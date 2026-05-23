#!/usr/bin/env python3
"""
Whisper Base 中文(台灣) LoRA 微調腳本

使用拼接後的 Common Voice zh-TW 資料集訓練。
支援 GPU (CUDA) 和 CPU 訓練。

用法:
  # GPU 訓練 (推薦)
  python train.py --use_gpu

  # CPU 訓練 (非常慢)
  python train.py

  # 自訂參數
  python train.py --use_gpu --batch_size 2 --epochs 5 --learning_rate 1e-4

  # 從 LoRA adapter 繼續訓練
  python train.py --use_gpu --resume_from_adapter ./whisper-base-zh-TW-lora --epochs 3

  # 從 Trainer checkpoint 繼續訓練
  python train.py --use_gpu --resume_from_checkpoint whisper-base-zh-TW-lora/checkpoint-XXX
"""

import os
import csv
import json
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np
import soundfile as sf
import torch
import evaluate
from datasets import Dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model

# ============================================================
# 配置
# ============================================================

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/whisper-base"
DEFAULT_LANGUAGE = "chinese"
DEFAULT_TASK = "transcribe"


# ============================================================
# 資料載入
# ============================================================

def load_concat_dataset(data_dir: str, split: str, clips_dir: str = None) -> Dataset:
    """載入拼接後的資料集"""
    tsv_path = os.path.join(data_dir, split, "data.tsv")
    if clips_dir is None:
        clips_dir = os.path.join(data_dir, split, "clips")

    rows = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append({
                "audio_path": os.path.join(clips_dir, row["path"]),
                "sentence": row["sentence"],
                "duration_ms": int(row["duration_ms"]),
                "num_clips": int(row["num_clips"]),
            })

    dataset = Dataset.from_list(rows)
    logger.info(f"Loaded {split} dataset: {len(dataset)} samples")
    return dataset


# ============================================================
# 音頻預載入
# ============================================================

def preload_audio_to_memory(dataset):
    """預先讀取所有音頻到記憶體，避免 map 時重複磁碟 I/O"""
    logger.info(f"Pre-loading {len(dataset)} audio files into memory...")
    audio_arrays = []
    sampling_rates = []

    for i, item in enumerate(dataset):
        audio_array, sr = sf.read(item["audio_path"], dtype="float32")

        # 立體聲轉單聲道
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        # 重取樣到 16kHz（安全檢查，我們的 WAV 已經是 16kHz）
        if sr != 16000:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
            sr = 16000

        audio_arrays.append(audio_array)
        sampling_rates.append(sr)

        if (i + 1) % 200 == 0:
            logger.info(f"  Loaded {i+1}/{len(dataset)} audio files")

    dataset = dataset.add_column("audio_array", audio_arrays)
    dataset = dataset.add_column("sampling_rate", sampling_rates)
    logger.info(f"Pre-loaded {len(dataset)} audio files into memory")
    return dataset


# ============================================================
# 資料預處理（批次處理，音頻已在記憶體）
# ============================================================

def prepare_dataset_batched(batch, feature_extractor, tokenizer):
    """批次處理：音頻已預載到 batch 中，直接計算 spectrogram + tokenize"""
    input_features = []
    labels = []

    for audio_array, sr, sentence in zip(
        batch["audio_array"], batch["sampling_rate"], batch["sentence"]
    ):
        # 計算 log-Mel spectrogram
        feat = feature_extractor(audio_array, sampling_rate=sr).input_features[0]
        input_features.append(feat)

        # 編碼文字標註
        label_ids = tokenizer(sentence).input_ids
        labels.append(label_ids)

    return {"input_features": input_features, "labels": labels}


# ============================================================
# Data Collator
# ============================================================

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """自定義 Data Collator，處理 Whisper 的 input_features 和 labels"""
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 分離 inputs 和 labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # 處理 labels
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # 用 -100 取代 padding，讓 loss 忽略這些位置
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # 移除 BOS token（如果有的話，因為訓練時會自動加上）
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ============================================================
# 評估指標
# ============================================================

def compute_metrics(pred, tokenizer, metric):
    """計算 WER (Word Error Rate)"""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # 將 -100 替換為 pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # 解碼
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # 計算 WER
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# ============================================================
# 主訓練流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Whisper Base 中文(台灣) LoRA 微調")
    parser.add_argument("--data_dir", type=str, default="./cv_zhTW_concat",
                        help="拼接後的資料集目錄")
    parser.add_argument("--output_dir", type=str, default="./whisper-base-zh-TW-lora",
                        help="模型輸出目錄")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL,
                        help="預訓練模型名稱或路徑")
    parser.add_argument("--language", type=str, default=DEFAULT_LANGUAGE,
                        help="語言設定")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK,
                        help="任務設定 (transcribe/translate)")
    parser.add_argument("--use_gpu", action="store_true",
                        help="使用 GPU 訓練")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="訓練 batch size (自動偵測)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None,
                        help="梯度累積步數 (自動偵測)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="訓練 epoch 數")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="學習率")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="預熱步數")
    parser.add_argument("--lora_r", type=int, default=32,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="凍結 encoder（加速訓練）")
    parser.add_argument("--max_label_length", type=int, default=256,
                        help="最大標籤長度")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="日誌記錄間隔")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="模型儲存間隔")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="評估間隔")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                        help="Early stopping 耐心值")
    parser.add_argument("--seed", type=int, default=42,
                        help="隨機種子")
    parser.add_argument("--fp16", action="store_true", default=None,
                        help="使用 FP16 混合精度訓練")
    parser.add_argument("--bf16", action="store_true", default=None,
                        help="使用 BF16 混合精度訓練")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=None,
                        help="使用梯度檢查點節省記憶體")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="從 Trainer checkpoint 目錄恢復訓練")
    parser.add_argument("--resume_from_adapter", type=str, default=None,
                        help="從 LoRA adapter 目錄恢復訓練（如 ./whisper-base-zh-TW-lora）")
    args = parser.parse_args()

    # --------------------------------------------------------
    # 設定日誌
    # --------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # --------------------------------------------------------
    # 偵測硬體環境
    # --------------------------------------------------------
    use_cuda = args.use_gpu and torch.cuda.is_available()
    if args.use_gpu and not torch.cuda.is_available():
        logger.warning("GPU requested but CUDA not available! Falling back to CPU.")

    if use_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"🟢 GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
    else:
        logger.info("🟡 Using CPU (training will be slow)")

    # 自動設定訓練參數
    if use_cuda:
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_mem_gb <= 4:
            # 4GB VRAM GPU (如 GTX 1050 Ti)
            batch_size = args.batch_size or 4
            grad_accum = args.gradient_accumulation_steps or 4
            fp16 = args.fp16 if args.fp16 is not None else True
            bf16 = args.bf16 if args.bf16 is not None else False
            grad_ckpt = args.gradient_checkpointing if args.gradient_checkpointing is not None else True
        elif gpu_mem_gb <= 8:
            batch_size = args.batch_size or 4
            grad_accum = args.gradient_accumulation_steps or 4
            fp16 = args.fp16 if args.fp16 is not None else True
            bf16 = args.bf16 if args.bf16 is not None else False
            grad_ckpt = args.gradient_checkpointing if args.gradient_checkpointing is not None else True
        else:
            batch_size = args.batch_size or 8
            grad_accum = args.gradient_accumulation_steps or 2
            fp16 = args.fp16 if args.fp16 is not None else True
            bf16 = args.bf16 if args.bf16 is not None else False
            grad_ckpt = args.gradient_checkpointing if args.gradient_checkpointing is not None else False
    else:
        # CPU
        batch_size = args.batch_size or 1
        grad_accum = args.gradient_accumulation_steps or 16
        fp16 = False
        bf16 = False
        grad_ckpt = args.gradient_checkpointing if args.gradient_checkpointing is not None else True

    logger.info(f"Training config:")
    logger.info(f"  batch_size={batch_size}, gradient_accumulation_steps={grad_accum}")
    logger.info(f"  fp16={fp16}, bf16={bf16}, gradient_checkpointing={grad_ckpt}")
    logger.info(f"  effective_batch_size={batch_size * grad_accum}")

    # --------------------------------------------------------
    # 載入模型和處理器
    # --------------------------------------------------------
    logger.info(f"Loading model: {args.model_name}")

    processor = WhisperProcessor.from_pretrained(
        args.model_name, language=args.language, task=args.task
    )
    tokenizer = WhisperTokenizer.from_pretrained(
        args.model_name, language=args.language, task=args.task
    )
    feature_extractor = processor.feature_extractor

    # 永遠以 FP32 載入模型，讓 Trainer 的 AMP 自動處理混合精度。
    # 若以 FP16/BF16 載入，LoRA 可訓練參數也會是半精度，
    # 導致 optimizer 儲存 FP16 gradients，GradScaler 無法正確 unscale。
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_name,
    )

    # 設定語言和任務
    model.generation_config.language = args.language
    model.generation_config.task = args.task
    model.generation_config.forced_decoder_ids = None

    # --------------------------------------------------------
    # 凍結 encoder（可選）
    # --------------------------------------------------------
    if args.freeze_encoder:
        logger.info("Freezing encoder...")
        model.freeze_encoder()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # 啟用梯度檢查點及 input require grads（必須在 LoRA 之前設定）
    if grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # --------------------------------------------------------
    # 設定 LoRA
    # --------------------------------------------------------
    if not args.freeze_encoder:
        logger.info("Setting up LoRA...")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
            modules_to_save=["proj_out"],
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # 從已有的 LoRA adapter 載入權重繼續訓練
        if args.resume_from_adapter:
            from peft import PeftModel
            logger.info(f"Loading adapter from {args.resume_from_adapter} for continued training...")
            adapter_config_path = os.path.join(args.resume_from_adapter, "adapter_config.json")
            if not os.path.exists(adapter_config_path):
                raise FileNotFoundError(
                    f"adapter_config.json not found in {args.resume_from_adapter}. "
                    "Please provide a valid LoRA adapter directory."
                )
            # 載入已有 adapter 的 config 來驗證相容性
            with open(adapter_config_path, "r") as f:
                saved_config = json.load(f)
            saved_targets = set(saved_config.get("target_modules", []))
            new_targets = set(lora_config.target_modules)
            if saved_targets != new_targets:
                logger.warning(
                    f"⚠️ Target modules mismatch! Saved: {saved_targets}, New: {new_targets}. "
                    "Incompatible adapters cannot be resumed. Using saved config."
                )
            # 用 is_trainable=True 載入，確保權重可繼續訓練
            model = PeftModel.from_pretrained(model, args.resume_from_adapter, is_trainable=True)
            model.print_trainable_parameters()
            logger.info("Adapter loaded successfully. Training will continue from this checkpoint.")

    # --------------------------------------------------------
    # 載入資料集
    # --------------------------------------------------------
    logger.info("Loading datasets...")
    train_dataset = load_concat_dataset(args.data_dir, "train")
    eval_dataset = load_concat_dataset(args.data_dir, "dev")

    logger.info(f"Train: {len(train_dataset)} samples")
    logger.info(f"Eval: {len(eval_dataset)} samples")

    # --------------------------------------------------------
    # 預載音頻到記憶體
    # --------------------------------------------------------
    train_dataset = preload_audio_to_memory(train_dataset)
    eval_dataset = preload_audio_to_memory(eval_dataset)

    # --------------------------------------------------------
    # 預處理資料集（批次處理，音頻已在記憶體）
    # --------------------------------------------------------
    logger.info("Preprocessing datasets (batched, audio in memory)...")

    train_dataset = train_dataset.map(
        lambda batch: prepare_dataset_batched(batch, feature_extractor, tokenizer),
        batched=True,
        batch_size=32,
        remove_columns=train_dataset.column_names,
        desc="Preprocessing train",
    )
    eval_dataset = eval_dataset.map(
        lambda batch: prepare_dataset_batched(batch, feature_extractor, tokenizer),
        batched=True,
        batch_size=32,
        remove_columns=eval_dataset.column_names,
        desc="Preprocessing eval",
    )

    logger.info(f"Processed train: {len(train_dataset)} samples")
    logger.info(f"Processed eval: {len(eval_dataset)} samples")

    # --------------------------------------------------------
    # Data Collator
    # --------------------------------------------------------
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # --------------------------------------------------------
    # 評估指標
    # --------------------------------------------------------
    wer_metric = evaluate.load("wer")

    def compute_metrics_fn(pred):
        return compute_metrics(pred, tokenizer, wer_metric)

    # --------------------------------------------------------
    # 訓練參數
    # --------------------------------------------------------
    training_args = Seq2SeqTrainingArguments(
        use_cpu=not use_cuda,
        output_dir=args.output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=grad_ckpt,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        logging_steps=args.logging_steps,
        logging_dir=os.path.join(args.output_dir, "logs"),
        report_to=["tensorboard"],
        predict_with_generate=True,
        generation_max_length=args.max_label_length,
        seed=args.seed,
        dataloader_num_workers=4 if use_cuda else 0,
        dataloader_pin_memory=use_cuda,
        remove_unused_columns=False,
        label_names=["labels"],
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        processing_class=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    # --------------------------------------------------------
    # 開始訓練
    # --------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Starting training!")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Dataset: {args.data_dir}")
    logger.info(f"  Train samples: {len(train_dataset)}")
    logger.info(f"  Eval samples: {len(eval_dataset)}")
    logger.info(f"  Batch size: {batch_size} x {grad_accum} = {batch_size * grad_accum} effective")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    logger.info(f"  Freeze encoder: {args.freeze_encoder}")
    logger.info(f"  FP16: {fp16}, BF16: {bf16}")
    logger.info(f"  Gradient checkpointing: {grad_ckpt}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info("=" * 60)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # --------------------------------------------------------
    # 儲存最終模型
    # --------------------------------------------------------
    logger.info("Saving final model...")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # 儲存訓練配置
    config = {
        "model_name": args.model_name,
        "language": args.language,
        "task": args.task,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "freeze_encoder": args.freeze_encoder,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "warmup_steps": args.warmup_steps,
        "fp16": fp16,
        "bf16": bf16,
        "gradient_checkpointing": grad_ckpt,
        "seed": args.seed,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
    }
    with open(os.path.join(args.output_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info(f"Model saved to {args.output_dir}")
    logger.info("Training complete! 🎉")


if __name__ == "__main__":
    main()