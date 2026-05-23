# Whisper Base 中文(台灣) LoRA 微調 - 部署指南

## 檔案結構

```
train_whisper_cpu/
├── train.py                  # 訓練腳本
├── concat_audio.py           # 音檔拼接腳本（已在本地執行過）
├── requirements.txt          # Python 依賴
├── setup_and_train.sh        # 遠端一鍵設定+訓練腳本
├── README.md                 # 本檔案
└── cv_zhTW_concat/           # 拼接後的資料集
    ├── train/
    │   ├── data.tsv
    │   └── clips/            # 981 個 WAV 檔案
    ├── dev/
    │   ├── data.tsv
    │   └── clips/            # 706 個 WAV 檔案
    └── test/
        ├── data.tsv
        └── clips/            # 771 個 WAV 檔案
```

## 遠端部署步驟

### 1. 打包資料

```powershell
# 在本機 Windows PowerShell 執行
cd d:\Codes\train_whisper_cpu

# 打包訓練腳本和資料集
# 注意：cv_zhTW_concat 約 2 GB
tar -czf whisper_train_package.tar.gz `
    train.py `
    requirements.txt `
    setup_and_train.sh `
    cv_zhTW_concat/
```

### 2. 上傳到遠端

```powershell
# 使用 scp 上傳（替換為你的遠端位址）
scp whisper_train_package.tar.gz user@remote-server:/path/to/train_whisper_cpu/

# 或使用 rsync（更快，支援斷點續傳）
rsync -avz --progress whisper_train_package.tar.gz user@remote-server:/path/to/train_whisper_cpu/
```

### 3. 在遠端解壓並訓練

```bash
# SSH 到遠端
ssh user@remote-server

# 解壓
cd /path/to/train_whisper_cpu
tar -xzf whisper_train_package.tar.gz

# 一鍵設定和訓練
chmod +x setup_and_train.sh
./setup_and_train.sh
```

### 4. 手動訓練（如果不想用一鍵腳本）

```bash
# 建立虛擬環境
python3 -m venv .venv
source .venv/bin/activate

# 安裝 PyTorch（根據 CUDA 版本）
# CUDA 12.1:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安裝其他依賴
pip install -r requirements.txt

# GPU 訓練（GTX 1050 Ti 4GB）
python train.py --use_gpu --batch_size 2 --gradient_accumulation_steps 8 --epochs 5

# CPU 訓練（非常慢）
python train.py --batch_size 1 --gradient_accumulation_steps 16 --epochs 3
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