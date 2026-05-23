# Copilot Instructions

## Project Overview

Fine-tuning OpenAI's Whisper-base model for Traditional Chinese (zh-TW) transcription using LoRA (via PEFT) on the Common Voice dataset. The pipeline concatenates short audio clips into ~30s segments to reduce padding waste during Whisper training, then trains with Hugging Face Seq2SeqTrainer.

## Architecture

Two-phase pipeline, each with its own script:

1. **Audio concatenation** (`concat_audio.py`) — Greedy-bins short Common Voice clips into groups totaling ~28s, inserts 0.5s silence gaps, joins text labels with `\n` (Whisper treats newlines as sentence breaks), outputs WAV files + `data.tsv` per split. Already run; output is in `cv_zhTW_concat/`.

2. **LoRA training** (`train.py`) — Loads concatenated dataset from `cv_zhTW_concat/`, pre-loads all audio into memory, computes log-Mel spectrograms + tokenizes labels in batched `map()`, trains with `Seq2SeqTrainer` + `EarlyStoppingCallback`, saves LoRA adapter to `whisper-base-zh-TW-lora/`.

`setup_and_train.sh` orchestrates both setup (venv, PyTorch, deps) and training in one shot with auto-detected GPU/CPU parameters.

`verify_concat.py` is a standalone validator for checking concatenated audio output (hardcoded Windows paths, not part of the main pipeline).

## Dataset Format

- Location: `cv_zhTW_concat/{train,dev,test}/`
- Each split has `data.tsv` (tab-separated) and `clips/` directory of WAV files
- TSV columns: `path`, `sentence`, `duration_ms`, `num_clips`
- Sentences with multiple original clips use `\n` as delimiter
- All audio: 16kHz mono float32 WAV

## Build & Run Commands

```bash
# Setup (creates .venv, installs PyTorch + deps)
chmod +x setup_and_train.sh && ./setup_and_train.sh

# Manual setup
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121  # adjust CUDA version
pip install -r requirements.txt

# GPU training
source .venv/bin/activate
python train.py --use_gpu --batch_size 2 --gradient_accumulation_steps 8 --epochs 5

# CPU training
source .venv/bin/activate
python train.py --batch_size 1 --gradient_accumulation_steps 16 --epochs 3

# Resume from checkpoint
python train.py --use_gpu --resume_from_checkpoint whisper-base-zh-TW-lora/checkpoint-XXX

# Audio concatenation (requires raw Common Voice corpus)
python concat_audio.py --input_dir <cv-corpus-path> --output_dir <output-path> --target_duration 28

# Monitor training
tensorboard --logdir ./whisper-base-zh-TW-lora/logs
```

## Key Conventions

- **LoRA target modules**: `q_proj` and `v_proj` with `modules_to_save=["proj_out"]` — do not change these without testing, as `proj_out` must be a full trainable module for language adaptation.
- **Language/task config**: WhisperProcessor and WhisperTokenizer are initialized with `language="chinese"`, `task="transcribe"`. The model's `generation_config` is also set to these values with `forced_decoder_ids=None`.
- **Auto-detection logic in train.py**: Batch size, gradient accumulation, fp16/bf16, and gradient checkpointing are auto-set based on detected VRAM (≤4GB, ≤8GB, >8GB). CLI args override auto-detection only when explicitly provided.
- **Audio pre-loading**: All audio is loaded into memory via `preload_audio_to_memory()` before `map()` preprocessing — necessary because the custom dataset doesn't use Hugging Face's built-in Audio feature.
- **Data collator**: `DataCollatorSpeechSeq2SeqWithPadding` strips BOS tokens from labels and replaces padding with -100 for loss masking. This is a custom implementation, not the standard `DataCollatorForSeq2Seq`.
- **WER metric**: Uses `evaluate` library's `wer` metric. `metric_for_best_model="wer"` with `greater_is_better=False` and `load_best_model_at_end=True`.
- **Label padding with -100**: Padding positions in labels are filled with `-100` so the cross-entropy loss ignores them.

## Environment Notes

- Python 3.12
- PyTorch must be installed separately (CUDA-version dependent) before `pip install -r requirements.txt`
- No test suite exists in this repository