#!/usr/bin/env python3
"""
使用 cv_zhTW_concat 測試集驗證 Whisper 模型

計算 WER (Word Error Rate) 和 CER (Character Error Rate)，
支援 CPU 模式（最準確）和 chunked ONNX 模式。

用法:
  # 驗證整個 test split（CPU 模式）
  python evaluate_model.py --cpu

  # 驗證前 N 筆
  python evaluate_model.py --cpu --max_samples 10

  # Chunked ONNX encoder 模式
  python evaluate_model.py --onnx

  # 使用自己的 LoRA adapter（不須先合併）
  python evaluate_model.py --cpu --adapter ./whisper-base-zh-TW-lora

  # 指定 merged 模型目錄
  python evaluate_model.py --cpu --model_dir ./whisper-base-zh-TW-merged
"""

import os
import sys
import csv
import argparse
import logging
import time
import numpy as np

logger = logging.getLogger(__name__)


def load_test_data(data_dir, split="test", max_samples=None):
    """載入 cv_zhTW_concat 測試集"""
    tsv_path = os.path.join(data_dir, split, "data.tsv")
    clips_dir = os.path.join(data_dir, split, "clips")

    samples = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)  # skip header
        for row in reader:
            if len(row) < 4:
                continue
            path, sentence, duration_ms, num_clips = row[0], row[1], row[2], row[3]
            audio_path = os.path.join(clips_dir, path)
            if not os.path.exists(audio_path):
                logger.warning(f"Audio not found: {audio_path}")
                continue
            samples.append({
                "audio_path": audio_path,
                "reference": sentence,
                "duration_ms": int(duration_ms),
                "num_clips": int(num_clips),
            })

    if max_samples:
        samples = samples[:max_samples]

    logger.info(f"Loaded {len(samples)} samples from {tsv_path}")
    return samples


def load_audio(audio_path, target_sr=16000):
    import soundfile as sf
    audio, sr = sf.read(audio_path, dtype="float32")
    if sr != target_sr:
        raise ValueError(f"Sample rate {sr} != {target_sr}")
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio


def transcribe_cpu(audio, model, processor, language="chinese", task="transcribe"):
    """純 CPU: 完整 30s HF pipeline"""
    import torch

    target_len = 30 * 16000
    if len(audio) > target_len:
        audio = audio[:target_len]
    else:
        audio = np.pad(audio, (0, target_len - len(audio)))

    mel = processor.feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features

    with torch.no_grad():
        predicted_ids = model.generate(
            mel, language=language, task=task, max_new_tokens=200,
        )

    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]


def transcribe_chunked(audio, model, processor, onnx_session,
                        language="chinese", task="transcribe"):
    """Chunked: 5s ONNX encoder + per-chunk HF decoder"""
    import torch
    from transformers.modeling_outputs import BaseModelOutput

    CHUNK_SAMPLES = 5 * 16000
    MEL_FRAMES = 500
    MAX_SOURCE = 1500
    D_MODEL = 512

    total_chunks = max(1, int(np.ceil(len(audio) / CHUNK_SAMPLES)))
    target_len = total_chunks * CHUNK_SAMPLES
    audio_padded = np.pad(audio, (0, target_len - len(audio)))

    texts = []
    for i in range(total_chunks):
        chunk = audio_padded[i * CHUNK_SAMPLES:(i + 1) * CHUNK_SAMPLES]
        mel = processor.feature_extractor(
            chunk, sampling_rate=16000, return_tensors="np"
        ).input_features[0][:, :MEL_FRAMES]
        mel_nhwc = mel.T[np.newaxis, np.newaxis, :, :].astype(np.float32)
        mel_nchw = mel_nhwc.transpose(0, 3, 1, 2)

        enc_out = onnx_session.run(None, {"mel_input": mel_nchw})[0]  # (1,250,512)

        padded = np.zeros((1, MAX_SOURCE, D_MODEL), dtype=np.float32)
        padded[:, :enc_out.shape[1], :] = enc_out

        fake_enc = BaseModelOutput(last_hidden_state=torch.from_numpy(padded))
        with torch.no_grad():
            ids = model.generate(
                encoder_outputs=fake_enc,
                language=language, task=task,
                max_new_tokens=100,
                no_repeat_ngram_size=3,
            )
        text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        if text:
            texts.append(text)

    return "\n".join(texts)


def load_model(model_dir=None, adapter_dir=None):
    """載入模型（支援 merged 模型或 LoRA adapter）"""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    if adapter_dir:
        logger.info(f"Loading base model + LoRA adapter: {adapter_dir}")
        base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, adapter_dir)
        model = model.merge_and_unload()
        processor = WhisperProcessor.from_pretrained(adapter_dir)
    else:
        logger.info(f"Loading merged model: {model_dir}")
        model = WhisperForConditionalGeneration.from_pretrained(model_dir)
        processor = WhisperProcessor.from_pretrained(model_dir)

    return model, processor


def normalize_chinese(text):
    """正規化中文文字"""
    text = text.replace("\n", "")  # 合併換行
    text = " ".join(text.split())  # 合併多餘空白
    return text.strip()


def char_level_normalize(text):
    """
    中文 WER 需要字級分詞：每個字之間加空格。
    jiwer 將無空格中文整串視為一個 word → WER 恆為 100%。
    在字間插入空格後 WER ≈ CER，但使用 word-level 編輯距離計算。
    """
    text = normalize_chinese(text)
    return " ".join(list(text))


def compute_wer_cer(references, hypotheses):
    """計算 WER 和 CER（中文需字級分詞計算 WER）"""
    try:
        import evaluate
        wer_metric = evaluate.load("wer")
        cer_metric = evaluate.load("cer")
    except ImportError:
        logger.error("Please install: pip install evaluate jiwer")
        sys.exit(1)

    # Normalize
    refs = [normalize_chinese(r) for r in references]
    hyps = [normalize_chinese(h) for h in hypotheses]

    # Filter empty
    pairs = [(r, h) for r, h in zip(refs, hyps) if r and h]
    if not pairs:
        return 100.0, 100.0

    refs_filtered, hyps_filtered = zip(*pairs)

    # CER: 直接用原始正規化文字
    cer = cer_metric.compute(references=refs_filtered, predictions=hyps_filtered)

    # WER: 中文需字級分詞，否則整句被視為單一 word
    refs_char = [char_level_normalize(r) for r in refs_filtered]
    hyps_char = [char_level_normalize(h) for h in hyps_filtered]
    wer = wer_metric.compute(references=refs_char, predictions=hyps_char)

    return wer * 100, cer * 100


def main():
    parser = argparse.ArgumentParser(description="驗證 Whisper 模型 (WER/CER)")
    parser.add_argument("--data_dir", default="./cv_zhTW_concat",
                        help="資料集目錄")
    parser.add_argument("--split", default="test",
                        help="資料集 split (test/dev)")
    parser.add_argument("--cpu", action="store_true",
                        help="純 CPU 模式（完整 30s HF pipeline）")
    parser.add_argument("--onnx", action="store_true",
                        help="Chunked ONNX encoder 模式")
    parser.add_argument("--adapter", default=None,
                        help="LoRA adapter 目錄（直接載入，不需先合併）")
    parser.add_argument("--model_dir", default="./hailo_export/whisper-merged-cpu",
                        help="Merged 模型目錄（--adapter 未指定時使用）")
    parser.add_argument("--onnx_path", default="./hailo_export/whisper_encoder.onnx",
                        help="Encoder ONNX 檔案")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最多驗證幾筆（debug 用）")
    parser.add_argument("--language", default="chinese", help="語言")
    parser.add_argument("--task", default="transcribe", help="任務")
    parser.add_argument("--show_details", action="store_true",
                        help="顯示每筆的參考/預測文字")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if not args.cpu and not args.onnx:
        args.cpu = True  # default to CPU mode

    # Load test data
    samples = load_test_data(args.data_dir, args.split, args.max_samples)

    # Load model
    model, processor = load_model(
        model_dir=args.model_dir, adapter_dir=args.adapter,
    )

    # Load ONNX session if needed
    onnx_session = None
    if args.onnx:
        import onnxruntime as ort
        onnx_session = ort.InferenceSession(args.onnx_path)
        logger.info(f"ONNX encoder loaded: {args.onnx_path}")

    # Evaluate
    references = []
    hypotheses = []
    total_duration = 0

    logger.info(f"Mode: {'CPU (30s)' if args.cpu else 'Chunked ONNX (5s)'}")
    logger.info(f"Evaluating {len(samples)} samples...")

    for i, sample in enumerate(samples):
        audio = load_audio(sample["audio_path"])
        ref = sample["reference"]
        total_duration += sample["duration_ms"] / 1000

        t0 = time.time()
        if args.cpu:
            hyp = transcribe_cpu(audio, model, processor, args.language, args.task)
        else:
            hyp = transcribe_chunked(audio, model, processor, onnx_session,
                                      args.language, args.task)
        dt = time.time() - t0

        ref_n = normalize_chinese(ref)
        hyp_n = normalize_chinese(hyp)

        references.append(ref_n)
        hypotheses.append(hyp_n)

        if args.show_details or i < 3:  # always show first 3
            logger.info(f"[{i+1}/{len(samples)}] ({sample['duration_ms']/1000:.1f}s, {dt:.1f}x)")
            logger.info(f"  REF: {ref_n[:80]}")
            logger.info(f"  HYP: {hyp_n[:80]}")

    # Compute metrics
    wer, cer = compute_wer_cer(references, hypotheses)

    print("\n" + "=" * 60)
    print(f"  Evaluation Results ({'CPU 30s' if args.cpu else 'Chunked ONNX 5s'})")
    print(f"  Samples: {len(samples)}")
    print(f"  Total audio: {total_duration:.1f}s")
    print("-" * 60)
    print(f"  WER: {wer:.2f}%")
    print(f"  CER: {cer:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()