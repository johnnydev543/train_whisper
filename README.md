# Whisper Base 中文(台灣) LoRA 微調

使用 LoRA (PEFT) 在 Common Voice 語料庫上微調 OpenAI Whisper-base，支援 Hailo-8 NPU 部署。

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
# 從 LoRA adapter 繼續訓練
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

### LoRA 目標模組

`q_proj`、`v_proj`、`k_proj`、`o_proj`，並設 `modules_to_save=["proj_out"]` 確保語言適應層可訓練。

### 預估訓練時間

| 硬體 | 每 epoch | 5 epochs |
|------|---------|----------|
| GTX 1050 Ti 4GB | ~30-50 分鐘 | ~2.5-4 小時 |
| RTX 3060 12GB | ~10-15 分鐘 | ~50-75 分鐘 |
| CPU (Ryzen 5 8600G) | ~5-8 小時 | ~25-40 小時 |

## 推論

```bash
# 純 CPU 模式（最準確，完整 30s encoder + decoder）
python hailo_inference.py audio.wav --cpu

# Chunked ONNX encoder + CPU decoder（5s 分塊編碼，測試 NPU encoder 輸出）
python hailo_inference.py audio.wav

# 使用 Hailo NPU encoder（需 Hailo-8 硬體）
python hailo_inference.py audio.wav --hef

# 驗證 encoder ONNX 輸出
python hailo_inference.py audio.wav --onnx_test
```

> **注意：** Chunked 模式將音頻切成 5s 片段各自編碼再解碼，因缺少跨片段 self-attention，準確度略低於完整 30s pipeline。`--cpu` 模式為最準確的參考基準。

## Hailo-8 NPU 部署

將 Whisper encoder 轉為 Hailo-8 HEF 格式，實現 NPU 加速推論。

### 前置需求

```bash
# 安裝系統依賴
sudo apt install libgraphviz-dev

# 安裝 Hailo Dataflow Compiler
pip install hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl

# 安裝 openai-whisper（export 腳本需要）
pip install openai-whisper onnxruntime
```

### 匯出 HEF

```bash
# 完整流程：合併 LoRA → 匯出 ONNX → 校準 → 編譯 HEF
python export_to_hailo.py

# 指定輸出目錄
python export_to_hailo.py --output_dir ./hailo_export

# 用真實音頻做 INT8 校準（推薦，量化更精確）
python export_to_hailo.py --calib_audio_dir ./cv_zhTW_concat/train/clips
```

### 匯出流程說明

1. **合併 LoRA 權重** → HF ↔ OpenAI Whisper 權重名稱對映
2. **Patch Whisper 原始碼** — Conv1d→Conv2d（Hailo 不支援 Conv1d）、停用 SDPA、縮短 audio context
3. **匯出 Encoder ONNX** — 輸入 (1,80,1,500)，輸出 (1,250,512)
4. **INT8 校準** — 產生 NHWC 格式校準資料
5. **編譯 HEF** — ~25 分鐘，輸出 ~21MB HEF 檔案

### 產出檔案

| 檔案 | 說明 |
|------|------|
| `whisper_encoder.onnx` | Encoder ONNX（測試用） |
| `whisper_encoder.hef` | Hailo-8 HEF（NPU 部署用） |
| `whisper-merged-cpu/` | 合併後完整 HF 模型（CPU decoder 推論用） |
| `encoder_calib.npy` | INT8 校準資料 |

### 架構

```
音頻 (30s)
  ├─ 5s chunk 1 → Hailo NPU encoder → HF CPU decoder → 轉錄 1
  ├─ 5s chunk 2 → Hailo NPU encoder → HF CPU decoder → 轉錄 2
  ├─ ...
  └─ 5s chunk 6 → Hailo NPU encoder → HF CPU decoder → 轉錄 6
                                                    ↓
                                            合併最終轉錄結果
```

Encoder 在 Hailo-8 NPU 上以 5s 片段執行，Decoder 在 CPU 上逐片段解碼。

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

### WER 不下降
- 嘗試降低學習率到 5e-5
- 增加 `--epochs` 到 10
- 增加 `--lora_r` 到 64
- 檢查資料品質

### 訓練太慢
- 確認 GPU 有被使用：`nvidia-smi`
- 確認 `--fp16` 有啟用（GPU 預設啟用）
- 減少 `--max_label_length` 到 128

### HEF 編譯失敗
- 確認已安裝 `libgraphviz-dev`
- 確認有足夠 RAM（建議 32GB，本專案 19GB 可用但較慢）
- 設定 `CUDA_VISIBLE_DEVICES=""` 避免 TensorFlow Conv2D 錯誤