#!/usr/bin/env python3
"""
Hailo-8 Whisper 推理 Pipeline

架構：
  - Encoder: Hailo NPU (HEF) / ONNX Runtime / HuggingFace (CPU)
  - Decoder: HuggingFace Transformers (CPU)

5s chunked encoding + per-chunk decoding pipeline:
  音頻 → 切 5s chunks → 每 chunk 獨立 encode → 每 chunk 獨立 decode → 合併轉錄

用法:
  # Chunked ONNX encoder + HF decoder（推薦，測試 Hailo encoder 輸出）
  python hailo_inference.py audio.wav

  # 純 CPU 模式（完整 30s encoder + decoder，最準確的參考基準）
  python hailo_inference.py audio.wav --cpu

  # 使用 Hailo HEF（需 Hailo-8 硬體 + runtime）
  python hailo_inference.py audio.wav --hef

  # ONNX encoder 單 chunk 測試
  python hailo_inference.py audio.wav --onnx_test
"""

import os
import sys
import argparse
import logging
import time
import numpy as np

logger = logging.getLogger(__name__)

CHUNK_SECONDS = 5
CHUNK_SAMPLES = CHUNK_SECONDS * 16000  # 80000
MEL_FRAMES_PER_CHUNK = 500  # whisper-base 5s → 500 mel frames
ENCODER_FRAMES_PER_CHUNK = 250  # after conv2 stride
D_MODEL = 512  # whisper-base hidden dim
MAX_SOURCE_POSITIONS = 1500  # 30s worth of encoder frames


def load_audio(audio_path, target_sr=16000):
    """載入音頻檔案"""
    import soundfile as sf
    audio, sr = sf.read(audio_path, dtype='float32')
    if sr != target_sr:
        raise ValueError(f"Sample rate {sr} != {target_sr}. Please resample first.")
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio


def audio_to_mel_chunk(audio_chunk, processor):
    """將 5s 音頻 chunk 轉為 mel spectrogram（NHWC 格式 for Hailo/ONNX）"""
    mel = processor.feature_extractor(
        audio_chunk, sampling_rate=16000, return_tensors="np"
    ).input_features[0][:, :MEL_FRAMES_PER_CHUNK]  # (80, 500)
    mel_nhwc = mel.T[np.newaxis, np.newaxis, :, :].astype(np.float32)  # (1,1,500,80)
    return mel_nhwc


def run_encoder_hailo(mel_nhwc, hef_path):
    """使用 Hailo NPU 運行 encoder（需 Hailo 硬體 + runtime）"""
    try:
        from hailo_platform import (VDevice, HailoStream,
                                     ConfigureParams, Hef, InferVStreams,
                                     InputVStreamParams, OutputVStreamparams)
    except ImportError:
        logger.error("hailo_platform not installed. Install Hailo runtime SDK.")
        sys.exit(1)

    hef = Hef(hef_path)
    params = ConfigureParams.create_from_hef(hef, interface=HailoStream.INTERFACE.PCIe)

    with VDevice() as vdevice:
        configure_group = vdevice.configure(hef, params)
        network_group = configure_group[0]
        input_vstream_params = InputVStreamParams.make(network_group)
        output_vstream_params = OutputVStreamparams.make(network_group)
        input_data = {name: mel_nhwc for name in input_vstream_params}

        with InferVStreams(network_group, input_vstream_params, output_vstream_params) as infer_pipeline:
            output = infer_pipeline.infer(input_data)

        output_key = list(output.keys())[0]
        return output[output_key]


def run_encoder_onnx(mel_nhwc, onnx_path):
    """使用 ONNX Runtime 運行 encoder（5s chunk）"""
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path)
    mel_nchw = mel_nhwc.transpose(0, 3, 1, 2)  # NHWC→NCHW: (1,80,1,500)
    return session.run(None, {"mel_input": mel_nchw})[0]


def run_chunk_decoder(encoder_output, model, processor, language, task):
    """
    對單個 chunk 的 encoder output 執行 decoder。

    encoder_output: (1, 250, 512) numpy array
    回傳: 轉錄文字字串
    """
    import torch
    from transformers.modeling_outputs import BaseModelOutput

    # 將 250-frame encoder output pad 到 1500（decoder 期望 30s 長度）
    # 非 padding 位置放真實 encoder 特徵，padding 用 0 填充
    padded = np.zeros((1, MAX_SOURCE_POSITIONS, D_MODEL), dtype=np.float32)
    n_frames = encoder_output.shape[1]
    padded[:, :n_frames, :] = encoder_output

    enc_tensor = torch.from_numpy(padded)
    fake_enc = BaseModelOutput(last_hidden_state=enc_tensor)

    with torch.no_grad():
        predicted_ids = model.generate(
            encoder_outputs=fake_enc,
            language=language,
            task=task,
            max_new_tokens=100,
            no_repeat_ngram_size=3,
            suppress_tokens=[],
        )

    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return text.strip()


def run_chunked_pipeline(audio, model_dir, encoder_fn, encoder_path,
                         language="chinese", task="transcribe"):
    """
    Chunked pipeline: 5s encoder chunks → per-chunk decoder → 合併。

    encoder_fn: 呼叫簽名 (mel_nhwc, path) → (1, 250, 512) numpy array
    """
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    logger.info(f"Loading model: {model_dir}")
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)
    processor = WhisperProcessor.from_pretrained(model_dir)

    # Pad audio to multiple of CHUNK_SECONDS
    total_chunks = max(1, int(np.ceil(len(audio) / CHUNK_SAMPLES)))
    target_len = total_chunks * CHUNK_SAMPLES
    audio_padded = np.pad(audio, (0, target_len - len(audio)))

    logger.info(f"  Audio: {len(audio)/16000:.1f}s → {total_chunks} chunks × {CHUNK_SECONDS}s")

    transcriptions = []
    t_start = time.time()

    for i in range(total_chunks):
        chunk_start = i * CHUNK_SAMPLES
        chunk_audio = audio_padded[chunk_start:chunk_start + CHUNK_SAMPLES]

        # Encode
        mel_nhwc = audio_to_mel_chunk(chunk_audio, processor)
        enc_out = encoder_fn(mel_nhwc, encoder_path)

        # Decode
        text = run_chunk_decoder(enc_out, model, processor, language, task)
        timestamp = f"[{i*CHUNK_SECONDS:02d}:{(i+1)*CHUNK_SECONDS:02d}]"
        logger.info(f"  Chunk {i+1}/{total_chunks} {timestamp}: {text}")
        transcriptions.append(text)

    elapsed = time.time() - t_start
    full_text = "\n".join(transcriptions)
    logger.info(f"  Inference time: {elapsed:.1f}s ({total_chunks*CHUNK_SECONDS/elapsed:.1f}x realtime)")

    return full_text


def run_full_pipeline_hf(audio, model_dir, language="chinese", task="transcribe"):
    """純 CPU 模式：完整 30s HF encoder + decoder（最準確的參考基準）"""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    logger.info(f"Loading model: {model_dir}")
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)
    processor = WhisperProcessor.from_pretrained(model_dir)

    target_len = 30 * 16000
    if len(audio) > target_len:
        audio = audio[:target_len]
    else:
        audio = np.pad(audio, (0, target_len - len(audio)))

    mel = processor.feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features

    logger.info(f"  Mel shape: {mel.shape}")
    logger.info("  Running full HF encoder + decoder...")

    t_start = time.time()
    with torch.no_grad():
        predicted_ids = model.generate(
            mel,
            language=language,
            task=task,
            max_new_tokens=200,
        )
    elapsed = time.time() - t_start

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    logger.info(f"  Inference time: {elapsed:.1f}s")
    return transcription


def main():
    parser = argparse.ArgumentParser(description="Hailo-8 Whisper 推理 Pipeline")
    parser.add_argument("audio", help="音頻檔案 (WAV 16kHz)")
    parser.add_argument("--hef_path", default="./hailo_export/whisper_encoder.hef",
                        help="Encoder HEF 檔案")
    parser.add_argument("--onnx_path", default="./hailo_export/whisper_encoder.onnx",
                        help="Encoder ONNX 檔案")
    parser.add_argument("--model_dir", default="./hailo_export/whisper-merged-cpu",
                        help="Whisper merged 模型目錄")
    parser.add_argument("--cpu", action="store_true",
                        help="純 CPU 模式（完整 30s HF pipeline，最準確）")
    parser.add_argument("--hef", action="store_true",
                        help="使用 Hailo NPU encoder（需硬體）")
    parser.add_argument("--onnx_test", action="store_true",
                        help="ONNX encoder 單 chunk 測試（驗證 encoder 輸出）")
    parser.add_argument("--language", default="chinese", help="語言")
    parser.add_argument("--task", default="transcribe", help="任務")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    audio = load_audio(args.audio)
    logger.info(f"Audio: {len(audio)/16000:.1f}s ({len(audio)} samples)")

    if args.cpu:
        transcription = run_full_pipeline_hf(
            audio, args.model_dir,
            language=args.language, task=args.task,
        )
    elif args.onnx_test:
        from transformers import WhisperProcessor
        processor = WhisperProcessor.from_pretrained(args.model_dir)

        chunk = audio[:CHUNK_SAMPLES] if len(audio) >= CHUNK_SAMPLES \
            else np.pad(audio, (0, CHUNK_SAMPLES - len(audio)))
        mel_nhwc = audio_to_mel_chunk(chunk, processor)

        encoder_output = run_encoder_onnx(mel_nhwc, args.onnx_path)
        logger.info(f"Encoder output: shape={encoder_output.shape}, "
                     f"mean={encoder_output.mean():.4f}, std={encoder_output.std():.4f}")
        print(f"\nONNX encoder test passed! Output shape: {encoder_output.shape}")
        return
    elif args.hef:
        if not os.path.exists(args.hef_path):
            logger.error(f"HEF not found: {args.hef_path}")
            sys.exit(1)
        transcription = run_chunked_pipeline(
            audio, args.model_dir,
            encoder_fn=run_encoder_hailo,
            encoder_path=args.hef_path,
            language=args.language, task=args.task,
        )
    else:
        # Default: chunked ONNX encoder + HF decoder
        if not os.path.exists(args.onnx_path):
            logger.error(f"ONNX not found: {args.onnx_path}")
            logger.info("Run export_to_hailo.py first, or use --cpu for pure HF pipeline")
            sys.exit(1)
        transcription = run_chunked_pipeline(
            audio, args.model_dir,
            encoder_fn=run_encoder_onnx,
            encoder_path=args.onnx_path,
            language=args.language, task=args.task,
        )

    print("\n" + "=" * 60)
    print("Transcription:")
    print("-" * 60)
    print(transcription)
    print("=" * 60)


if __name__ == "__main__":
    main()