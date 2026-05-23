#!/usr/bin/env python3
"""
Whisper → Hailo-8 HEF 轉換腳本

架構策略：
  - Encoder → HEF（在 Hailo NPU 上運行）
  - Decoder → CPU（使用 HuggingFace Transformers，因為沒有 KV-cache NPU 會慢 12 倍）

流程:
  1. Patch OpenAI whisper source (Conv1d→Conv2d, disable SDPA, scale n_audio_ctx)
  2. 合併 HF LoRA → OpenAI whisper 權重
  3. 匯出 Encoder ONNX
  4. 產生 Encoder 校正集（NHWC 格式）
  5. 儲存 Decoder 完整模型（CPU 推理用）
  6. ONNX → HEF (encoder only)

用法:
  # 完整流程
  python export_to_hailo.py

  # 只匯出 ONNX
  python export_to_hailo.py --skip_hef

  # 用真實音頻做校正集
  python export_to_hailo.py --calib_audio_dir ./cv_zhTW_concat/train/clips
"""

import os
import sys
import argparse
import logging
import shutil
import re
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
from onnxsim import simplify

logger = logging.getLogger(__name__)


# ============================================================
# Step 1: Patch OpenAI whisper source
# ============================================================

def _find_whisper_model_py():
    """找到 whisper model.py 的路徑"""
    import whisper
    import inspect
    return inspect.getfile(whisper.model)


def patch_whisper_source(scaling_factor=6):
    """
    Patch OpenAI whisper model.py:
    - Conv1d → Conv2d
    - SDPA_AVAILABLE = False
    - n_audio_ctx // scaling_factor
    - flatten(2) in encoder forward
    - Value projection * 1.0 fix
    """
    model_py = _find_whisper_model_py()
    backup_path = model_py + ".backup"

    with open(model_py, 'r') as f:
        source = f.read()

    if not os.path.exists(backup_path):
        shutil.copy2(model_py, backup_path)
        logger.info(f"  Backed up: {backup_path}")

    # 1. Disable SDPA
    source = source.replace("SDPA_AVAILABLE = True", "SDPA_AVAILABLE = False  # Hailo")

    # 2. Conv1d → Conv2d class
    source = source.replace("class Conv1d(nn.Conv1d):", "class Conv2d(nn.Conv2d):")

    # 3. Encoder conv layers
    source = source.replace(
        "self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)",
        "self.conv1 = Conv2d(n_mels, n_state, kernel_size=(1, 3), padding=(0, 1))"
    )
    source = source.replace(
        "self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)",
        "self.conv2 = Conv2d(n_state, n_state, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))"
    )

    # 4. flatten(2) before permute in encoder forward
    source = source.replace(
        "x = x.permute(0, 2, 1)",
        "x = x.flatten(2).permute(0, 2, 1)"
    )

    # 5. Scale n_audio_ctx
    source = re.sub(
        r'(self\.dims\.n_audio_ctx\s*//\s*)\d+',
        rf'\g<1>{scaling_factor}',
        source
    )

    # 6. Value projection * 1.0 fix
    source = source.replace(
        "v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)",
        "v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * 1.0"
    )

    with open(model_py, 'w') as f:
        f.write(source)

    logger.info(f"  Patched: {model_py}")
    return model_py


def restore_whisper_source():
    """還原原始 whisper model.py"""
    model_py = _find_whisper_model_py()
    backup_path = model_py + ".backup"
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, model_py)
        os.remove(backup_path)
        logger.info(f"  Restored: {model_py}")


# ============================================================
# Step 2: Merge HF LoRA → OpenAI whisper
# ============================================================

def merge_and_load(model_name, adapter_dir, scaling_factor=6):
    """合併 HF LoRA，轉換權重到 OpenAI whisper 格式"""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from peft import PeftModel

    logger.info(f"Loading HF base model: {model_name}")
    hf_model = WhisperForConditionalGeneration.from_pretrained(model_name)
    logger.info(f"Loading LoRA adapter: {adapter_dir}")
    peft_model = PeftModel.from_pretrained(hf_model, adapter_dir)
    logger.info("Merging LoRA weights...")
    hf_merged = peft_model.merge_and_unload()
    hf_sd = hf_merged.state_dict()

    # Reload patched whisper
    import whisper
    import importlib
    importlib.reload(whisper.model)
    importlib.reload(whisper)

    dims = whisper.model.ModelDimensions(
        n_mels=80,
        n_audio_ctx=1500 // scaling_factor,
        n_audio_state=512, n_audio_head=8, n_audio_layer=6,
        n_vocab=51865, n_text_ctx=448, n_text_state=512,
        n_text_head=8, n_text_layer=4,
    )
    oa_model = whisper.model.Whisper(dims)

    # Build HF→OA name mapping
    mapping = _build_weight_mapping()
    new_sd = {}
    for oa_name in oa_model.state_dict():
        hf_keys = [k for k, v in mapping.items() if v == oa_name]
        if not hf_keys:
            continue
        for hf_key in hf_keys:
            if hf_key in hf_sd:
                weight = hf_sd[hf_key].clone()
                # Conv1d→Conv2d: unsqueeze dim=2
                if "conv1.weight" in oa_name or "conv2.weight" in oa_name:
                    weight = weight.unsqueeze(2)
                # Trim encoder positional embedding
                if "positional_embedding" in oa_name and "encoder" in oa_name:
                    weight = weight[:1500 // scaling_factor]
                new_sd[oa_name] = weight
                break

    missing, unexpected = oa_model.load_state_dict(new_sd, strict=False)
    real_missing = [m for m in missing if 'mask' not in m and 'alignment' not in m]
    if real_missing:
        logger.warning(f"  Missing weights: {real_missing}")

    oa_model.eval()
    logger.info("  Loaded merged weights into patched whisper model")

    processor = WhisperProcessor.from_pretrained(adapter_dir)
    return oa_model, processor


def _build_weight_mapping():
    """HF → OpenAI 權重名稱對應表"""
    m = {
        "model.encoder.conv1.weight": "encoder.conv1.weight",
        "model.encoder.conv1.bias": "encoder.conv1.bias",
        "model.encoder.conv2.weight": "encoder.conv2.weight",
        "model.encoder.conv2.bias": "encoder.conv2.bias",
        "model.encoder.embed_positions.weight": "encoder.positional_embedding",
        "model.encoder.layer_norm.weight": "encoder.ln_post.weight",
        "model.encoder.layer_norm.bias": "encoder.ln_post.bias",
        "model.decoder.embed_tokens.weight": "decoder.token_embedding.weight",
        "model.decoder.embed_positions.weight": "decoder.positional_embedding",
        "model.decoder.layer_norm.weight": "decoder.ln.weight",
        "model.decoder.layer_norm.bias": "decoder.ln.bias",
    }
    # Encoder layers (6 for base)
    for i in range(6):
        _add_layer_map(m, f"model.encoder.layers.{i}", f"encoder.blocks.{i}")
    # Decoder layers (4 for base)
    for i in range(4):
        _add_layer_map(m, f"model.decoder.layers.{i}", f"decoder.blocks.{i}", cross=True)
    return m


def _add_layer_map(m, hf, oa, cross=False):
    """Add attention/MLP weight mapping for one layer"""
    m[f"{hf}.self_attn.q_proj.weight"] = f"{oa}.attn.query.weight"
    m[f"{hf}.self_attn.q_proj.bias"] = f"{oa}.attn.query.bias"
    m[f"{hf}.self_attn.k_proj.weight"] = f"{oa}.attn.key.weight"
    m[f"{hf}.self_attn.v_proj.weight"] = f"{oa}.attn.value.weight"
    m[f"{hf}.self_attn.v_proj.bias"] = f"{oa}.attn.value.bias"
    m[f"{hf}.self_attn.out_proj.weight"] = f"{oa}.attn.out.weight"
    m[f"{hf}.self_attn.out_proj.bias"] = f"{oa}.attn.out.bias"
    m[f"{hf}.self_attn_layer_norm.weight"] = f"{oa}.attn_ln.weight"
    m[f"{hf}.self_attn_layer_norm.bias"] = f"{oa}.attn_ln.bias"
    if cross:
        m[f"{hf}.encoder_attn.q_proj.weight"] = f"{oa}.cross_attn.query.weight"
        m[f"{hf}.encoder_attn.q_proj.bias"] = f"{oa}.cross_attn.query.bias"
        m[f"{hf}.encoder_attn.k_proj.weight"] = f"{oa}.cross_attn.key.weight"
        m[f"{hf}.encoder_attn.v_proj.weight"] = f"{oa}.cross_attn.value.weight"
        m[f"{hf}.encoder_attn.v_proj.bias"] = f"{oa}.cross_attn.value.bias"
        m[f"{hf}.encoder_attn.out_proj.weight"] = f"{oa}.cross_attn.out.weight"
        m[f"{hf}.encoder_attn.out_proj.bias"] = f"{oa}.cross_attn.out.bias"
        m[f"{hf}.encoder_attn_layer_norm.weight"] = f"{oa}.cross_attn_ln.weight"
        m[f"{hf}.encoder_attn_layer_norm.bias"] = f"{oa}.cross_attn_ln.bias"
    m[f"{hf}.fc1.weight"] = f"{oa}.mlp.0.weight"
    m[f"{hf}.fc1.bias"] = f"{oa}.mlp.0.bias"
    m[f"{hf}.fc2.weight"] = f"{oa}.mlp.2.weight"
    m[f"{hf}.fc2.bias"] = f"{oa}.mlp.2.bias"
    m[f"{hf}.final_layer_norm.weight"] = f"{oa}.mlp_ln.weight"
    m[f"{hf}.final_layer_norm.bias"] = f"{oa}.mlp_ln.bias"


# ============================================================
# Step 3: Export Encoder ONNX
# ============================================================

def export_encoder_onnx(model, output_dir, audio_chunk_seconds=5):
    """匯出 Encoder ONNX"""
    mel_frames = audio_chunk_seconds * 100
    encoder_path = os.path.join(output_dir, "whisper_encoder.onnx")
    dummy_input = torch.randn(1, 80, 1, mel_frames)

    logger.info(f"Exporting encoder ONNX: {encoder_path}")
    logger.info(f"  Input: {dummy_input.shape}")

    torch.onnx.export(
        model.encoder, dummy_input, encoder_path,
        opset_version=14,
        input_names=["mel_input"],
        output_names=["encoder_hidden_states"],
    )

    # Simplify
    model_onnx = onnx.load(encoder_path)
    model_simp, check = simplify(model_onnx)
    if check:
        onnx.save(model_simp, encoder_path)
        logger.info("  Simplified successfully")

    # Verify
    model_onnx = onnx.load(encoder_path)
    onnx.checker.check_model(model_onnx)
    inp = model_onnx.graph.input[0].type.tensor_type.shape
    out = model_onnx.graph.output[0].type.tensor_type.shape
    logger.info(f"  Verified: input={_dims(inp)}, output={_dims(out)}")

    # Check op types
    ops = sorted(set(n.op_type for n in model_onnx.graph.node))
    logger.info(f"  Op types: {ops}")

    return encoder_path


def _dims(shape):
    return tuple(int(d.dim_value) for d in shape.dim)


# ============================================================
# Step 4: Calibration
# ============================================================

def create_encoder_calibration(output_dir, processor=None, calib_audio_dir=None,
                               num_samples=2048, audio_chunk_seconds=5):
    """產生 Encoder NHWC 校正集"""
    mel_frames = audio_chunk_seconds * 100
    logger.info(f"Creating encoder calibration ({num_samples} samples)...")
    calib_data = []

    if calib_audio_dir and processor:
        import soundfile as sf
        audio_files = sorted(f for f in os.listdir(calib_audio_dir) if f.endswith('.wav'))
        for af in audio_files[:num_samples]:
            try:
                audio, sr = sf.read(os.path.join(calib_audio_dir, af), dtype='float32')
                if sr != 16000:
                    continue
                target_len = audio_chunk_seconds * 16000
                if len(audio) >= target_len:
                    audio = audio[:target_len]
                else:
                    audio = np.pad(audio, (0, target_len - len(audio)))
                mel = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="np").input_features[0]
                mel = mel[:, :mel_frames]  # (80, 500)
                mel_nhwc = mel.T[np.newaxis, np.newaxis, :, :]  # (1, 1, 500, 80)
                calib_data.append(mel_nhwc.astype(np.float32))
            except Exception as e:
                logger.debug(f"  Skip {af}: {e}")
        logger.info(f"  Generated {len(calib_data)} samples from real audio")
    else:
        logger.info("  Using random mel spectrograms")
        for _ in range(num_samples):
            mel = np.random.randn(1, 1, mel_frames, 80).astype(np.float32) * 1.5
            calib_data.append(mel)

    if not calib_data:
        logger.error("No calibration data!")
        return None

    calib_array = np.concatenate(calib_data, axis=0)
    calib_path = os.path.join(output_dir, "encoder_calib.npy")
    np.save(calib_path, calib_array)
    logger.info(f"  Saved: {calib_path} shape={calib_array.shape}")
    return calib_path


# ============================================================
# Step 5: Save decoder for CPU inference
# ============================================================

def save_decoder_assets(model_name, adapter_dir, output_dir):
    """儲存 decoder 推理用的完整 merged HF 模型"""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from peft import PeftModel

    merged_path = os.path.join(output_dir, "whisper-merged-cpu")
    if os.path.exists(os.path.join(merged_path, "model.safetensors")):
        logger.info(f"  Merged CPU model exists: {merged_path}")
        processor = WhisperProcessor.from_pretrained(merged_path)
        return merged_path, processor

    logger.info(f"Saving merged model for CPU inference: {merged_path}")
    hf_model = WhisperForConditionalGeneration.from_pretrained(model_name)
    peft_model = PeftModel.from_pretrained(hf_model, adapter_dir)
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(merged_path)
    processor = WhisperProcessor.from_pretrained(adapter_dir)
    processor.save_pretrained(merged_path)
    logger.info(f"  Saved: {merged_path}")
    return merged_path, processor


# ============================================================
# Step 6: Compile HEF
# ============================================================

def compile_encoder_hef(encoder_onnx_path, output_dir, hw_arch="hailo8",
                        calib_path=None):
    """編譯 Encoder ONNX → HEF"""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from hailo_sdk_client import ClientRunner

    har_path = os.path.join(output_dir, "whisper_encoder.har")
    hef_path = os.path.join(output_dir, "whisper_encoder.hef")

    logger.info("Compiling encoder HEF...")
    runner = ClientRunner(hw_arch=hw_arch)
    runner.translate_onnx_model(encoder_onnx_path, "whisper_encoder")

    if calib_path and os.path.exists(calib_path):
        calib_data = np.load(calib_path)
        logger.info(f"  Quantizing with {len(calib_data)} samples...")
        runner.optimize(calib_data=calib_data)

    runner.save_har(har_path)
    logger.info(f"  HAR saved: {har_path}")

    logger.info("  Compiling HEF (this may take 10+ minutes)...")
    hef = runner.compile()
    with open(hef_path, 'wb') as f:
        f.write(hef)

    size_mb = os.path.getsize(hef_path) / 1024 / 1024
    logger.info(f"  HEF saved: {hef_path} ({size_mb:.1f} MB)")
    return hef_path


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Whisper → Hailo-8 HEF 轉換 (Encoder only, Decoder on CPU)")
    parser.add_argument("--adapter_dir", default="./whisper-base-zh-TW-lora")
    parser.add_argument("--model_name", default="openai/whisper-base")
    parser.add_argument("--output_dir", default="./hailo_export")
    parser.add_argument("--hw_arch", default="hailo8", choices=["hailo8", "hailo8l", "hailo15h"])
    parser.add_argument("--audio_seconds", type=int, default=5, help="音頻長度（秒）")
    parser.add_argument("--num_calib", type=int, default=2048, help="校正集樣本數")
    parser.add_argument("--calib_audio_dir", default=None, help="校正用音頻目錄")
    parser.add_argument("--skip_hef", action="store_true", help="只匯出 ONNX")
    parser.add_argument("--skip_onnx", action="store_true", help="跳過 ONNX 匯出")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    os.makedirs(args.output_dir, exist_ok=True)

    scaling_factor = 30 // args.audio_seconds  # 5s→6

    # Step 1: Patch
    logger.info("Step 1: Patching OpenAI whisper...")
    try:
        patch_whisper_source(scaling_factor=scaling_factor)
    except Exception as e:
        logger.error(f"Patch failed: {e}")
        return

    try:
        # Step 2: Merge & load
        logger.info("Step 2: Merging LoRA weights...")
        model, processor = merge_and_load(args.model_name, args.adapter_dir, scaling_factor)

        # Step 3: Export encoder ONNX
        if not args.skip_onnx:
            logger.info("Step 3: Exporting encoder ONNX...")
            encoder_onnx_path = export_encoder_onnx(model, args.output_dir, args.audio_seconds)
        else:
            encoder_onnx_path = os.path.join(args.output_dir, "whisper_encoder.onnx")
            logger.info(f"Step 3: Using existing ONNX: {encoder_onnx_path}")

        # Step 4: Calibration
        logger.info("Step 4: Creating calibration set...")
        calib_path = create_encoder_calibration(
            args.output_dir, processor, args.calib_audio_dir,
            args.num_calib, args.audio_seconds,
        )

        # Step 5: Save decoder for CPU inference
        logger.info("Step 5: Saving decoder (CPU inference model)...")
        merged_path, _ = save_decoder_assets(args.model_name, args.adapter_dir, args.output_dir)

        if args.skip_hef:
            logger.info("=" * 60)
            logger.info("ONNX export complete!")
            logger.info(f"  Encoder ONNX: {encoder_onnx_path}")
            logger.info(f"  Encoder calib: {calib_path}")
            logger.info(f"  Decoder (CPU): {merged_path}")
            logger.info("")
            logger.info("Next: python export_to_hailo.py --skip_onnx (to compile HEF)")
            logger.info("   or: python export_to_hailo.py (full pipeline)")
            logger.info("=" * 60)
            return

        # Step 6: Compile HEF
        logger.info("Step 6: Compiling encoder HEF...")
        encoder_hef = compile_encoder_hef(encoder_onnx_path, args.output_dir,
                                            args.hw_arch, calib_path)

        logger.info("=" * 60)
        logger.info("Export complete!")
        logger.info(f"  Encoder HEF: {encoder_hef}")
        logger.info(f"  Decoder (CPU): {merged_path}")
        logger.info("")
        logger.info("Deployment: python hailo_inference.py <audio.wav>")
        logger.info("=" * 60)

    finally:
        logger.info("Restoring whisper source...")
        restore_whisper_source()


if __name__ == "__main__":
    main()