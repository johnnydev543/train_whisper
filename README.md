# Whisper Base 中文(台灣) LoRA 微調

## 快速開始

```bash
# 一鍵設定環境 + 訓練
chmod +x setup_and_train.sh && ./setup_and_train.sh

# 或手動設定
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121  # 依 CUDA 版本調整
pip install -r requirements.txt

# GPU 訓練（參數自動偵測）
python train.py --use_gpu

# CPU 訓練（非常慢）
python train.py
```

## 繼續訓練

```bash
# 從 LoRA adapter 繼續訓練（最常用）
python train.py --use_gpu --resume_from_adapter ./whisper-base-zh-TW-lora --epochs 3

# 從 Trainer checkpoint 繼續訓練
python train.py --use_gpu --resume_from_checkpoint whisper-base-zh-TW-lora/checkpoint-300
```

## 訓練參數說明

### GTX 1050 Ti 4GB 推薦參數

| 參數 | 值 | 說明 |
|------|-----|------|
| `--use_gpu` | ✅ | 啟用 GPU |
| `--batch_size` | 4 | 自動偵測 |
| `--gradient_accumulation_steps` | 4 | 有效 batch size = 4×4 = 16 |
| `--epochs` | 5 | 訓練 5 個 epoch |
| `--learning_rate` | 1e-4 | LoRA 用較大學習率 |
| `--lora_r` | 32 | LoRA rank |
| `--lora_alpha` | 64 | LoRA alpha |
| `--lora_dropout` | 0.05 | LoRA dropout |
| `--freeze_encoder` | 不建議 | LoRA 已足夠輕量 |

### 預估訓練時間

| 硬體 | 每 epoch | 5 epochs |
|------|---------|----------|
| GTX 1050 Ti 4GB | ~30-50 分鐘 | ~2.5-4 小時 |
| RTX 3060 12GB | ~10-15 分鐘 | ~50-75 分鐘 |
| CPU (Ryzen 5 8600G) | ~5-8 小時 | ~25-40 小時 |

## 訓練後使用模型

### 合併 LoRA 權重

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

# 載入基礎模型
base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
# 載入 LoRA 權重
model = PeftModel.from_pretrained(base_model, "./whisper-base-zh-TW-lora")
# 合併權重
model = model.merge_and_unload()
# 儲存合併後的模型
model.save_pretrained("./whisper-base-zh-TW-merged")
processor = WhisperProcessor.from_pretrained("./whisper-base-zh-TW-lora")
processor.save_pretrained("./whisper-base-zh-TW-merged")
```

### 推論

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torchaudio

# 載入模型
model = WhisperForConditionalGeneration.from_pretrained("./whisper-base-zh-TW-merged")
processor = WhisperProcessor.from_pretrained("./whisper-base-zh-TW-merged")

# 載入音頻
waveform, sr = torchaudio.load("test.wav")
if sr != 16000:
    waveform = torchaudio.functional.resample(waveform, sr, 16000)

# 推論
input_features = processor(
    waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt"
).input_features

predicted_ids = model.generate(input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print(f"Transcription: {transcription}")
```

## 監控訓練

```bash
# 啟動 TensorBoard
tensorboard --logdir ./whisper-base-zh-TW-lora/logs

# 在瀏覽器開啟
# 本機: http://localhost:6006
# 遠端: 使用 SSH 隧道
#   ssh -L 6006:localhost:6006 user@remote-server
```

## 常見問題

### OOM (Out of Memory)
- 降低 `--batch_size` 到 1
- 增加 `--gradient_accumulation_steps` 到 16
- 確保 `--gradient_checkpointing` 已啟用（預設啟用）
- 使用 `--freeze_encoder` 凍結 encoder

### 繼續訓練

```bash
# 從 LoRA adapter 繼續訓練（最常用）
python train.py --use_gpu --resume_from_adapter ./whisper-base-zh-TW-lora --epochs 3

# 從 Trainer checkpoint 繼續訓練
python train.py --use_gpu --resume_from_checkpoint whisper-base-zh-TW-lora/checkpoint-300
```

### WER 不下降
- 嘗試降低學習率到 5e-5
- 增加 `--epochs` 到 10
- 增加 `--lora_r` 到 64
- 檢查資料品質

### 訓練太慢
- 確認 GPU 有被使用：`nvidia-smi`
- 確認 `--fp16` 有啟用（GPU 預設啟用）
- 減少 `--max_label_length` 到 128