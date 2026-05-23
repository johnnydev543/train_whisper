#!/usr/bin/env python3
"""
將 Common Voice 短音檔拼接成接近 30 秒的片段，以減少 Whisper 訓練時的 padding 浪費。

策略：
- 將同一個 split 中的短音檔按順序拼接，直到總時長接近 30 秒
- 拼接的音檔之間加入 0.5 秒靜音間隔
- 文字標註用換行符（\n）連接，Whisper 會將其視為多句
- 輸出為新的 TSV 檔案和拼接後的 WAV 檔案

用法：
  python concat_audio.py --input_dir <cv-corpus-path> --output_dir <output-path> --target_duration 28 --silence_gap 0.5
"""

import os
import csv
import argparse
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import soundfile as sf


def load_tsv(tsv_path: str) -> tuple[list[str], list[dict]]:
    """載入 TSV 檔案，回傳 header 和資料行"""
    rows = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            row_dict = {}
            for i, col in enumerate(header):
                row_dict[col] = row[i] if i < len(row) else ""
            rows.append(row_dict)
    return header, rows


def load_durations(durations_path: str) -> dict[str, int]:
    """載入 clip_durations.tsv，回傳 {filename: duration_ms}"""
    durations = {}
    with open(durations_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            durations[row[0]] = int(row[1])
    return durations


def concat_clips(
    clip_groups: list[list[dict]],
    clips_dir: str,
    target_duration_ms: int,
    silence_gap_ms: int = 500,
    sample_rate: int = 16000,
) -> list[dict]:
    """
    將音檔群組拼接成接近 target_duration 的片段。

    回傳每個拼接片段的資訊：
    - audio_array: 拼接後的音頻陣列
    - duration_ms: 拼接後的總時長
    - sentence: 拼接後的文字標註
    - source_clips: 原始音檔資訊列表
    """
    results = []

    for group in clip_groups:
        audio_arrays = []
        sentence_parts = []
        total_duration_ms = 0
        source_clips = []

        silence_gap = np.zeros(int(silence_gap_ms * sample_rate / 1000), dtype=np.float32)

        for i, clip in enumerate(group):
            clip_path = os.path.join(clips_dir, clip["path"])

            # 讀取音頻
            audio, sr = sf.read(clip_path, dtype="float32")

            # 重取樣到 16kHz 如果需要
            if sr != sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

            # 如果是立體聲，轉為單聲道
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            audio_arrays.append(audio)
            sentence_parts.append(clip["sentence"])
            source_clips.append(clip)
            total_duration_ms += clip["duration_ms"]

            # 在音檔之間加入靜音間隔（最後一個不加）
            if i < len(group) - 1:
                audio_arrays.append(silence_gap)

        # 拼接所有音頻
        concatenated = np.concatenate(audio_arrays) if audio_arrays else np.array([], dtype=np.float32)

        # 用換行符連接文字
        combined_sentence = "\n".join(sentence_parts)

        results.append({
            "audio_array": concatenated,
            "duration_ms": total_duration_ms + (len(group) - 1) * silence_gap_ms,
            "actual_duration_ms": int(len(concatenated) / sample_rate * 1000),
            "sentence": combined_sentence,
            "source_clips": source_clips,
            "num_clips": len(group),
        })

    return results


def group_clips_by_duration(
    clips: list[dict],
    durations: dict[str, int],
    target_duration_ms: int,
    max_duration_ms: int = 30000,
) -> list[list[dict]]:
    """
    將短音檔分組，每組總時長接近 target_duration_ms。

    使用貪婪算法：依序將音檔加入當前組，直到加入下一個會超過 max_duration_ms。
    """
    groups = []
    current_group = []
    current_duration = 0

    # 靜音間隔的時長（每個間隔 500ms）
    silence_gap_ms = 500

    for clip in clips:
        clip_duration = durations.get(clip["path"], 0)

        # 如果單個音檔就超過 max_duration，單獨一組
        if clip_duration >= max_duration_ms:
            if current_group:
                groups.append(current_group)
                current_group = []
                current_duration = 0
            groups.append([clip])
            continue

        # 計算加入這個音檔後的總時長（包含靜音間隔）
        additional_ms = clip_duration + (silence_gap_ms if current_group else 0)

        if current_duration + additional_ms <= max_duration_ms:
            current_group.append(clip)
            current_duration += additional_ms
        else:
            # 當前組已滿，開始新組
            if current_group:
                groups.append(current_group)
            current_group = [clip]
            current_duration = clip_duration

    # 最後一組
    if current_group:
        groups.append(current_group)

    return groups


def process_split(
    split_name: str,
    tsv_path: str,
    durations_path: str,
    clips_dir: str,
    output_dir: str,
    target_duration_ms: int = 28000,
    max_duration_ms: int = 30000,
    silence_gap_ms: int = 500,
    sample_rate: int = 16000,
):
    """處理一個 split（train/dev/test）"""
    print(f"\n{'='*60}")
    print(f"Processing {split_name} split...")
    print(f"{'='*60}")

    # 載入資料
    header, rows = load_tsv(tsv_path)
    durations = load_durations(durations_path)

    # 加入 duration 資訊到每個 clip
    for row in rows:
        row["duration_ms"] = durations.get(row["path"], 0)

    # 過濾掉沒有 duration 的 clip
    valid_rows = [row for row in rows if row["duration_ms"] > 0]
    skipped = len(rows) - len(valid_rows)
    if skipped > 0:
        print(f"  Skipped {skipped} clips without duration info")

    print(f"  Total valid clips: {len(valid_rows)}")
    print(f"  Total duration: {sum(r['duration_ms'] for r in valid_rows)/1000/3600:.2f} hours")

    # 分組
    groups = group_clips_by_duration(valid_rows, durations, target_duration_ms, max_duration_ms)

    # 統計
    group_sizes = [len(g) for g in groups]
    group_durations = [sum(c["duration_ms"] for c in g) for g in groups]

    print(f"  Number of concatenated clips: {len(groups)}")
    print(f"  Avg clips per group: {np.mean(group_sizes):.1f}")
    print(f"  Avg group duration: {np.mean(group_durations)/1000:.2f}s")
    print(f"  Min group duration: {min(group_durations)/1000:.2f}s")
    print(f"  Max group duration: {max(group_durations)/1000:.2f}s")

    # 建立輸出目錄
    output_clips_dir = os.path.join(output_dir, split_name, "clips")
    os.makedirs(output_clips_dir, exist_ok=True)

    # 拼接音檔
    concatenated = concat_clips(groups, clips_dir, target_duration_ms, silence_gap_ms, sample_rate)

    # 寫入音檔和 TSV
    output_tsv_path = os.path.join(output_dir, split_name, "data.tsv")
    tsv_rows = []

    for i, item in enumerate(concatenated):
        # 儲存音頻檔案
        audio_filename = f"concat_{split_name}_{i:06d}.wav"
        audio_path = os.path.join(output_clips_dir, audio_filename)
        sf.write(audio_path, item["audio_array"], sample_rate)

        # TSV 行
        tsv_rows.append({
            "path": audio_filename,
            "sentence": item["sentence"],
            "duration_ms": str(item["actual_duration_ms"]),
            "num_clips": str(item["num_clips"]),
        })

    # 寫入 TSV
    with open(output_tsv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "sentence", "duration_ms", "num_clips"], delimiter="\t")
        writer.writeheader()
        writer.writerows(tsv_rows)

    print(f"  Saved {len(concatenated)} concatenated clips to {output_clips_dir}")
    print(f"  Saved TSV to {output_tsv_path}")

    # 統計拼接後的時長分佈
    actual_durations = [item["actual_duration_ms"] for item in concatenated]
    bins = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000]
    counts = [0] * (len(bins) - 1)
    for d in actual_durations:
        for j in range(len(bins) - 1):
            if bins[j] <= d < bins[j + 1]:
                counts[j] += 1
                break

    print(f"\n  Concatenated duration distribution:")
    for j in range(len(bins) - 1):
        bar = "#" * (counts[j] // 5)
        print(f"    {bins[j]/1000:.0f}-{bins[j+1]/1000:.0f}s: {counts[j]:>5}  {bar}")

    return len(concatenated)


def main():
    parser = argparse.ArgumentParser(description="拼接 Common Voice 短音檔成接近 30 秒的片段")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=r"d:\Codes\train_whisper_cpu\1774205381984-cv-corpus-25.0-2026-03-09-zh-TW\cv-corpus-25.0-2026-03-09\zh-TW",
        help="Common Voice zh-TW 資料目錄",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"d:\Codes\train_whisper_cpu\cv_zhTW_concat",
        help="輸出目錄",
    )
    parser.add_argument(
        "--target_duration",
        type=int,
        default=28,
        help="目標拼接時長（秒），預設 28 秒（留 2 秒緩衝）",
    )
    parser.add_argument(
        "--max_duration",
        type=int,
        default=30,
        help="最大允許時長（秒），預設 30 秒",
    )
    parser.add_argument(
        "--silence_gap",
        type=float,
        default=0.5,
        help="音檔間靜音間隔（秒），預設 0.5 秒",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="輸出取樣率，預設 16000（Whisper 預設）",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    clips_dir = os.path.join(input_dir, "clips")
    durations_path = os.path.join(input_dir, "clip_durations.tsv")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target duration: {args.target_duration}s")
    print(f"Max duration: {args.max_duration}s")
    print(f"Silence gap: {args.silence_gap}s")
    print(f"Sample rate: {args.sample_rate}Hz")

    total_concatenated = 0
    for split in ["train", "dev", "test"]:
        tsv_path = os.path.join(input_dir, f"{split}.tsv")
        if not os.path.exists(tsv_path):
            print(f"Skipping {split}: {tsv_path} not found")
            continue

        count = process_split(
            split_name=split,
            tsv_path=tsv_path,
            durations_path=durations_path,
            clips_dir=clips_dir,
            output_dir=output_dir,
            target_duration_ms=args.target_duration * 1000,
            max_duration_ms=args.max_duration * 1000,
            silence_gap_ms=int(args.silence_gap * 1000),
            sample_rate=args.sample_rate,
        )
        total_concatenated += count

    print(f"\n{'='*60}")
    print(f"Done! Total concatenated clips: {total_concatenated}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()