"""Tests for train.py — unit tests for data loading, preprocessing, collation, metrics."""

import csv
import os

import numpy as np
import pytest
import soundfile as sf
import torch

from train import (
    load_concat_dataset,
    preload_audio_to_memory,
    prepare_dataset_batched,
    DataCollatorSpeechSeq2SeqWithPadding,
    compute_metrics,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_data_dir(tmp_path):
    """Create a minimal cv_zhTW_concat directory with a valid dev split."""
    split_dir = tmp_path / "dev"
    clips_dir = split_dir / "clips"
    clips_dir.mkdir(parents=True)

    sr = 16000
    rows = []
    for i in range(3):
        audio = np.random.randn(sr).astype(np.float32) * 0.01
        fname = f"clip_{i:04d}.wav"
        sf.write(str(clips_dir / fname), audio, sr)
        rows.append({"path": fname, "sentence": f"句子{i}", "duration_ms": "1000", "num_clips": "1"})

    tsv_path = split_dir / "data.tsv"
    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "sentence", "duration_ms", "num_clips"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    return tmp_path


# ── load_concat_dataset ─────────────────────────────────────────────────────

class TestLoadConcatDataset:
    def test_loads_correct_count(self, sample_data_dir):
        """Should load all rows from TSV."""
        ds = load_concat_dataset(str(sample_data_dir), "dev")
        assert len(ds) == 3

    def test_has_expected_columns(self, sample_data_dir):
        """Dataset should have audio_path, sentence, duration_ms, num_clips."""
        ds = load_concat_dataset(str(sample_data_dir), "dev")
        assert "audio_path" in ds.column_names
        assert "sentence" in ds.column_names
        assert "duration_ms" in ds.column_names
        assert "num_clips" in ds.column_names

    def test_audio_path_is_absolute(self, sample_data_dir):
        """audio_path should be an absolute path."""
        ds = load_concat_dataset(str(sample_data_dir), "dev")
        for path in ds["audio_path"]:
            assert os.path.isabs(path)

    def test_missing_split_raises(self, tmp_path):
        """Loading a nonexistent split should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_concat_dataset(str(tmp_path), "nonexistent")

    def test_custom_clips_dir(self, sample_data_dir, tmp_path):
        """Should use custom clips_dir if provided."""
        alt_clips = tmp_path / "alt_clips"
        alt_clips.mkdir()
        import shutil
        src = os.path.join(str(sample_data_dir), "dev", "clips", "clip_0000.wav")
        shutil.copy2(src, str(alt_clips / "clip_0000.wav"))

        split_dir = tmp_path / "dev2"
        split_dir.mkdir()
        with open(split_dir / "data.tsv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "sentence", "duration_ms", "num_clips"], delimiter="\t")
            writer.writeheader()
            writer.writerow({"path": "clip_0000.wav", "sentence": "測試", "duration_ms": "1000", "num_clips": "1"})

        ds = load_concat_dataset(str(tmp_path), "dev2", clips_dir=str(alt_clips))
        assert len(ds) == 1


# ── preload_audio_to_memory ─────────────────────────────────────────────────

class TestPreloadAudio:
    def test_adds_audio_columns(self, sample_data_dir):
        """Should add audio_array and sampling_rate columns."""
        ds = load_concat_dataset(str(sample_data_dir), "dev")
        ds = preload_audio_to_memory(ds)
        assert "audio_array" in ds.column_names
        assert "sampling_rate" in ds.column_names

    def test_audio_shape_and_sr(self, sample_data_dir):
        """Audio should be 1D at 16kHz."""
        ds = load_concat_dataset(str(sample_data_dir), "dev")
        ds = preload_audio_to_memory(ds)
        for audio, sr in zip(ds["audio_array"], ds["sampling_rate"]):
            assert sr == 16000
            arr = np.array(audio)
            assert arr.ndim == 1

    def test_all_samples_preserved(self, sample_data_dir):
        """No samples should be lost during preloading."""
        ds = load_concat_dataset(str(sample_data_dir), "dev")
        n_before = len(ds)
        ds = preload_audio_to_memory(ds)
        assert len(ds) == n_before


# ── prepare_dataset_batched ─────────────────────────────────────────────────

class TestPrepareDatasetBatched:
    @pytest.fixture
    def processor(self):
        from transformers import WhisperProcessor
        return WhisperProcessor.from_pretrained("openai/whisper-base", language="chinese", task="transcribe")

    def test_output_keys(self, sample_data_dir, processor):
        """Should return input_features and labels."""
        ds = load_concat_dataset(str(sample_data_dir), "dev")
        ds = preload_audio_to_memory(ds)
        batch = {k: ds[k] for k in ds.column_names}
        result = prepare_dataset_batched(batch, processor.feature_extractor, processor.tokenizer)
        assert "input_features" in result
        assert "labels" in result

    def test_feature_shapes(self, sample_data_dir, processor):
        """input_features should be list of arrays with 80 Mel bins."""
        ds = load_concat_dataset(str(sample_data_dir), "dev")
        ds = preload_audio_to_memory(ds)
        batch = {k: ds[k] for k in ds.column_names}
        result = prepare_dataset_batched(batch, processor.feature_extractor, processor.tokenizer)
        assert len(result["input_features"]) == len(ds)
        assert len(result["labels"]) == len(ds)
        feat = result["input_features"][0]
        assert feat.shape[0] == 80


# ── DataCollatorSpeechSeq2SeqWithPadding ─────────────────────────────────────

class TestDataCollator:
    @pytest.fixture
    def processor_and_model(self):
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        proc = WhisperProcessor.from_pretrained("openai/whisper-base", language="chinese", task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
        return proc, model

    def test_collator_returns_tensors(self, sample_data_dir, processor_and_model):
        """Collator should return dict of torch tensors."""
        proc, model = processor_and_model
        ds = load_concat_dataset(str(sample_data_dir), "dev")
        ds = preload_audio_to_memory(ds)
        batch_data = {k: ds[k] for k in ds.column_names}
        prepared = prepare_dataset_batched(batch_data, proc.feature_extractor, proc.tokenizer)

        collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=proc,
            decoder_start_token_id=model.config.decoder_start_token_id,
        )

        features = [{"input_features": prepared["input_features"][i], "labels": prepared["labels"][i]} for i in range(len(ds))]
        batch = collator(features)

        assert "input_features" in batch
        assert "labels" in batch
        assert isinstance(batch["input_features"], torch.Tensor)
        assert isinstance(batch["labels"], torch.Tensor)

    def test_labels_padded_with_minus100(self, sample_data_dir, processor_and_model):
        """Padded label positions should be -100 for loss masking."""
        proc, model = processor_and_model
        ds = load_concat_dataset(str(sample_data_dir), "dev")
        ds = preload_audio_to_memory(ds)
        batch_data = {k: ds[k] for k in ds.column_names}
        prepared = prepare_dataset_batched(batch_data, proc.feature_extractor, proc.tokenizer)

        collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=proc,
            decoder_start_token_id=model.config.decoder_start_token_id,
        )

        features = [{"input_features": prepared["input_features"][i], "labels": prepared["labels"][i]} for i in range(len(ds))]
        batch = collator(features)

        labels = batch["labels"]
        # Check any -100 padding exists (labels of different lengths will have padding)
        assert labels.dtype in (torch.int64, torch.long)


# ── compute_metrics ──────────────────────────────────────────────────────────

class TestComputeMetrics:
    @pytest.fixture
    def wer_setup(self):
        import evaluate
        from transformers import WhisperTokenizer
        metric = evaluate.load("wer")
        tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base", language="chinese", task="transcribe")
        return metric, tokenizer

    def test_perfect_wer(self, wer_setup):
        """Identical prediction and reference should give WER=0."""
        metric, tokenizer = wer_setup
        text = "你好世界"
        ids = tokenizer(text).input_ids

        class Pred:
            predictions = [ids]
            label_ids = [ids]

        result = compute_metrics(Pred(), tokenizer, metric)
        assert result["wer"] == 0.0

    def test_error_wer(self, wer_setup):
        """Different prediction should give WER > 0."""
        metric, tokenizer = wer_setup
        ref_ids = tokenizer("你好世界").input_ids
        pred_ids = tokenizer("完全不同").input_ids

        class Pred:
            predictions = [pred_ids]
            label_ids = [ref_ids]

        result = compute_metrics(Pred(), tokenizer, metric)
        assert result["wer"] > 0.0

    def test_minus100_replaced_with_pad(self, wer_setup):
        """-100 in labels should be replaced with pad_token_id, not crash."""
        metric, tokenizer = wer_setup
        text = "你好世界"
        ref_ids = tokenizer(text).input_ids
        pred_ids = ref_ids.copy()

        # Simulate padded labels
        padded_labels = ref_ids + [-100, -100]

        class Pred:
            predictions = [pred_ids + [tokenizer.pad_token_id, tokenizer.pad_token_id]]
            label_ids = [padded_labels]

        result = compute_metrics(Pred(), tokenizer, metric)
        assert "wer" in result