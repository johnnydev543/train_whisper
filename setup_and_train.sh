#!/bin/bash
# ============================================================
# Whisper Base 中文(台灣) LoRA 微調 - 遠端 Linux 環境設定
# 
# 用法:
#   chmod +x setup_and_train.sh
#   ./setup_and_train.sh
# ============================================================

set -e

# ---- 配置 ----
PYTHON_VERSION="3.12"
WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${WORK_DIR}/.venv"
DATA_DIR="${WORK_DIR}/cv_zhTW_concat"
OUTPUT_DIR="${WORK_DIR}/whisper-base-zh-TW-lora"

# CUDA 版本（修改為你的 CUDA 版本）
# 選項: cu118, cu121, cu124
CUDA_VERSION="cu121"

# ---- 顏色輸出 ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
echo_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
echo_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ---- 步驟 1: 檢查 Python ----
echo_info "Checking Python ${PYTHON_VERSION}..."
if command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
    echo_info "Found Python: $(python3 --version)"
else
    echo_error "Python3 not found! Please install Python ${PYTHON_VERSION}"
    exit 1
fi

# ---- 步驟 2: 檢查 CUDA ----
echo_info "Checking CUDA..."
if command -v nvidia-smi &> /dev/null; then
    echo_info "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    HAS_CUDA=1
else
    echo_warn "No GPU detected. Will use CPU (very slow)."
    HAS_CUDA=0
fi

# ---- 步驟 3: 建立虛擬環境 ----
echo_info "Creating virtual environment..."
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
    echo_info "Virtual environment created at ${VENV_DIR}"
else
    echo_info "Virtual environment already exists"
fi

source "${VENV_DIR}/bin/activate"
echo_info "Activated virtual environment"

# ---- 步驟 4: 安裝 PyTorch ----
echo_info "Installing PyTorch..."
if [ "${HAS_CUDA}" -eq 1 ]; then
    echo_info "Installing PyTorch with CUDA ${CUDA_VERSION}..."
    pip install --upgrade pip
    pip install torch torchaudio --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"
else
    echo_info "Installing PyTorch (CPU only)..."
    pip install --upgrade pip
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# ---- 步驟 5: 安裝其他依賴 ----
echo_info "Installing dependencies..."
pip install -r "${WORK_DIR}/requirements.txt"

# ---- 步驟 6: 驗證安裝 ----
echo_info "Verifying installation..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
import transformers
print(f'Transformers: {transformers.__version__}')
import peft
print(f'PEFT: {peft.__version__}')
"

# ---- 步驟 7: 檢查資料 ----
echo_info "Checking dataset..."
if [ -d "${DATA_DIR}" ]; then
    echo_info "Dataset found at ${DATA_DIR}"
    for split in train dev test; do
        if [ -f "${DATA_DIR}/${split}/data.tsv" ]; then
            count=$(wc -l < "${DATA_DIR}/${split}/data.tsv")
            echo_info "  ${split}: $((count - 1)) samples"
        fi
    done
else
    echo_error "Dataset not found at ${DATA_DIR}!"
    echo_error "Please upload the cv_zhTW_concat directory first."
    exit 1
fi

# ---- 步驟 8: 開始訓練 ----
echo_info "Starting training..."
echo_info "========================================"

if [ "${HAS_CUDA}" -eq 1 ]; then
    echo_info "Training with GPU"
    python3 "${WORK_DIR}/train.py" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --use_gpu \
        --epochs 5 \
        --learning_rate 1e-4 \
        --warmup_steps 50 \
        --lora_r 32 \
        --lora_alpha 64 \
        --lora_dropout 0.05 \
        --save_steps 50 \
        --eval_steps 50 \
        --logging_steps 10 \
        --early_stopping_patience 5 \
        --seed 42
else
    echo_info "Training with CPU (this will be slow)"
    python3 "${WORK_DIR}/train.py" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --epochs 3 \
        --learning_rate 5e-5 \
        --warmup_steps 20 \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_dropout 0.1 \
        --save_steps 100 \
        --eval_steps 100 \
        --logging_steps 5 \
        --early_stopping_patience 3 \
        --seed 42
fi

echo_info "========================================"
echo_info "Training complete! 🎉"
echo_info "Model saved to: ${OUTPUT_DIR}"

# ---- 步驟 9: 測試模型 ----
echo_info "Running quick test..."
python3 -c "
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch

model_path = '${OUTPUT_DIR}'
processor = WhisperProcessor.from_pretrained(model_path)

# 載入模型（合併 LoRA 權重）
from peft import PeftModel
base_model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-base')
model = PeftModel.from_pretrained(base_model, model_path)
model = model.merge_and_unload()

# 測試推論
import numpy as np
# 生成 5 秒靜音作為測試
dummy_audio = np.zeros(16000 * 5, dtype=np.float32)
input_features = processor.feature_extractor(dummy_audio, sampling_rate=16000).input_features[0]
input_features = torch.tensor([input_features])

# 如果有 GPU 就用 GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
input_features = input_features.to(device)

with torch.no_grad():
    predicted_ids = model.generate(input_features)

transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print(f'Test transcription (should be empty/silence): \"{transcription}\"')
print('Model loaded and inference successful!')
"