#!/usr/bin/env python3
"""
Hailo-8 Whisper 推理 Pipeline

架構：
  - Encoder: Hailo NPU (HEF) 或 ONNX Runtime 或 HuggingFace
  - Decoder: CPU (HuggingFace Transformers)

注意：HEF encoder 使用 5s chunks，但 Whisper decoder 原生需要 30s encoder output。
因此目前 --cpu 模式使用完整 HF encoder+decoder（30s），確保正確性。
HEF 模式需 Hailo 硬體，deploy 時搭配 Hailo runtime。

用法:
  # 純 CPU 模式（完整 30s encoder + decoder，最準確）
  python hailo_inference.py audio.wav --cpu

  # 使用 Hailo HEF（需硬體）
  python hailo_inference.py audio.wav --hef ./hailo_export/whisper_encoder.hef
"""

import os
import sys
import argparse
import logging
import numpy as np

logger = logging.getLogger(__name__)


def load_audio(audio_path, target_sr=16000):
    """載入音頻檔案"""
    import soundfile as sf
    audio, sr = sf.read(audio_path, dtype='float32')
    if sr != target_sr:
        raise ValueError(f"Sample rate {sr} != {target_sr}. Please resample first.")
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio


def run_encoder_hailo(mel_nhwc, hef_path):
    """使用 Hailo NPU 運行 encoder（需 Hailo 硬體 + runtime）"""
    try:
        from hailo_platform import (VDevice, HailoStream,
                                     ConfigureParams, Hef, InferVStreams,
                                     InputVStreamParams, OutputVStreamparams)
    except ImportError:
        logger.error("hailo_platform not installed. Install Hailo runtime.")
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
    """使用 ONNX Runtime 運行 encoder（5s HEF 版本，測試用）"""
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path)
    # NHWC (1,1,500,80) → NCHW (1,80,1,500)
    mel_nchw = mel_nhwc.transpose(0, 3, 1, 2)
    return session.run(None, {"mel_input": mel_nchw})[0]


def run_full_pipeline_hf(audio, model_dir, language="chinese", task="transcribe"):
    """使用 HuggingFace 完整 pipeline（30s encoder + decoder）"""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    logger.info(f"Loading model: {model_dir}")
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)
    processor = WhisperProcessor.from_pretrained(model_dir)

    # 處理音頻（30s）
    target_len = 30 * 16000
    if len(audio) > target_len:
        audio = audio[:target_len]
    else:
        audio = np.pad(audio, (0, target_len - len(audio)))

    mel = processor.feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features

    logger.info(f"  Mel shape: {mel.shape}")
    logger.info("  Running encoder + decoder (full HF pipeline)...")

    with torch.no_grad():
        predicted_ids = model.generate(
            mel,
            language=language,
            task=task,
            max_new_tokens=200,
        )

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription


def main():
    parser = argparse.ArgumentParser(description="Hailo-8 Whisper Inference")
    parser.add_argument("audio", help="音頻檔案 (WAV 16kHz)")
    parser.add_argument("--hef", default="./hailo_export/whisper_encoder.hef",
                        help="Encoder HEF 檔案（需 Hailo 硬體）")
    parser.add_argument("--onnx", default="./hailo_export/whisper_encoder.onnx",
                        help="Encoder ONNX 檔案（測試用）")
    parser.add_argument("--model_dir", default="./hailo_export/whisper-merged-cpu",
                        help="Whisper merged 模型目錄")
    parser.add_argument("--cpu", action="store_true",
                        help="純 CPU 模式（完整 HF pipeline，最準確）")
    parser.add_argument("--onnx_test", action="store_true",
                        help="用 ONNX Runtime 測試 encoder 輸出")
    parser.add_argument("--language", default="chinese", help="語言")
    parser.add_argument("--task", default="transcribe", help="任務")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    audio = load_audio(args.audio)
    logger.info(f"Audio: {len(audio)/16000:.1f}s")

    if args.cpu:
        # 純 CPU 模式：完整 HF pipeline（30s encoder + decoder）
        transcription = run_full_pipeline_hf(
            audio, args.model_dir,
            language=args.language, task=args.task,
        )
    elif args.onnx_test:
        # ONNX 測試模式：只用 ONNX Runtime 跑 encoder，驗證輸出
        import torch
        from transformers import WhisperProcessor

        processor = WhisperProcessor.from_pretrained(args.model_dir)
        target_len = 5 * 16000
        chunk = audio[:target_len] if len(audio) >= target_len else np.pad(audio, (0, target_len - len(audio)))

        mel = processor.feature_extractor(chunk, sampling_rate=16000, return_tensors="np").input_features[0]
        mel = mel[:, :500]  # (80, 500)
        mel_nhwc = mel.T[np.newaxis, np.newaxis, :, :].astype(np.float32)

        encoder_output = run_encoder_onnx(mel_nhwc, args.onnx)
        logger.info(f"Encoder output: shape={encoder_output.shape}, "
                     f"mean={encoder_output.mean():.4f}, std={encoder_output.std():.4f}")
        print("\nONNX encoder test passed!")
        print(f"  Output shape: {encoder_output.shape}")
        print("  Note: This is a 5s chunk encoder. For full transcription, use --cpu mode.")
        return
    else:
        # Hailo NPU 模式
        if not os.path.exists(args.hef):
            logger.error(f"HEF not found: {args.hef}")
            logger.info("Use --cpu for full CPU pipeline or --onnx_test for encoder testing")
            sys.exit(1)

        from transformers import WhisperProcessor
        processor = WhisperProcessor.from_pretrained(args.model_dir)

        # 將長音頻分 5s chunks
        chunk_samples = 5 * 16000
        chunks = []
        for start in range(0, len(audio), chunk_samples):
            chunk = audio[start:start + chunk_samples]
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            if len(chunk) < chunk_samples:
                continue
            mel = processor.feature_extractor(
                chunk, sampling_rate=16000, return_tensors="np"
            ).input_features[0][:, :500]
            mel_nhwc = mel.T[np.newaxis, np.newaxis, :, :].astype(np.float32)
            chunks.append(mel_nhwc)

        logger.info(f"  {len(chunks)} audio chunks")

        # 跑每個 chunk 的 encoder
        all_encoder_outputs = []
        for i, mel_nhwc in enumerate(chunks):
            logger.info(f"  Running encoder chunk {i+1}/{len(chunks)}...")
            enc_out = run_encoder_hailo(mel_nhwc, args.hef)
            all_encoder_outputs.append(enc_out)

        # 注意：Hailo encoder (5s chunks) 輸出與 HF decoder (30s) 不直接相容
        # 部署時需要實作分塊 decoder 或修改 decoder 輸入
        logger.warning("Hailo 5s encoder chunks need chunk-aware decoder (not yet implemented)")
        logger.info("Use --cpu for accurate full-pipeline transcription")
        return

    print("\n" + "=" * 60)
    print(f"Transcription: {transcription}")
    print("=" * 60)


if __name__ == "__main__":
    main()