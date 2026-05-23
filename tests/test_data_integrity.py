"""Integration tests — verify the actual concatenated dataset in cv_zhTW_concat/."""

import csv
import os

import numpy as np
import pytest
import soundfile as sf


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cv_zhTW_concat")


def _tsv_rows(split):
    """Load data.tsv rows for a given split."""
    tsv_path = os.path.join(DATA_DIR, split, "data.tsv")
    if not os.path.exists(tsv_path):
        pytest.skip(f"{split} data.tsv not found")
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


# ── TSV structure ────────────────────────────────────────────────────────────

class TestTSVStructure:
    @pytest.mark.parametrize("split", ["train", "dev", "test"])
    def test_tsv_exists(self, split):
        """data.tsv should exist for all splits."""
        path = os.path.join(DATA_DIR, split, "data.tsv")
        assert os.path.exists(path), f"Missing {path}"

    @pytest.mark.parametrize("split", ["train", "dev", "test"])
    def test_tsv_columns(self, split):
        """TSV should have required columns: path, sentence, duration_ms, num_clips."""
        rows = _tsv_rows(split)
        assert len(rows) > 0
        for row in rows:
            assert "path" in row
            assert "sentence" in row
            assert "duration_ms" in row
            assert "num_clips" in row

    @pytest.mark.parametrize("split", ["train", "dev", "test"])
    def test_positive_durations(self, split):
        """All duration_ms values should be positive integers."""
        rows = _tsv_rows(split)
        for row in rows:
            dur = int(row["duration_ms"])
            assert dur > 0, f"Non-positive duration for {row['path']}"

    @pytest.mark.parametrize("split", ["train", "dev", "test"])
    def test_positive_num_clips(self, split):
        """All num_clips values should be >= 1."""
        rows = _tsv_rows(split)
        for row in rows:
            n = int(row["num_clips"])
            assert n >= 1, f"num_clips < 1 for {row['path']}"

    @pytest.mark.parametrize("split", ["train", "dev", "test"])
    def test_sentences_not_empty(self, split):
        """No sentence should be empty."""
        rows = _tsv_rows(split)
        for row in rows:
            assert row["sentence"].strip(), f"Empty sentence for {row['path']}"


# ── Audio files ──────────────────────────────────────────────────────────────

class TestAudioFiles:
    @pytest.mark.parametrize("split", ["train", "dev", "test"])
    def test_clips_dir_exists(self, split):
        """clips/ directory should exist."""
        clips_dir = os.path.join(DATA_DIR, split, "clips")
        assert os.path.isdir(clips_dir), f"Missing clips dir: {clips_dir}"

    @pytest.mark.parametrize("split", ["train", "dev", "test"])
    def test_all_tsv_files_exist(self, split):
        """Every audio file referenced in TSV should exist on disk."""
        rows = _tsv_rows(split)
        clips_dir = os.path.join(DATA_DIR, split, "clips")
        missing = [r["path"] for r in rows if not os.path.exists(os.path.join(clips_dir, r["path"]))]
        assert not missing, f"Missing {len(missing)} audio files in {split}: {missing[:5]}"

    @pytest.mark.parametrize("split", ["dev", "test"])
    def test_audio_format(self, split):
        """Audio files should be 16kHz mono float32 WAV (sample check)."""
        rows = _tsv_rows(split)
        clips_dir = os.path.join(DATA_DIR, split, "clips")
        indices = [0, len(rows) // 2, -1]
        for idx in indices:
            row = rows[idx]
            audio, sr = sf.read(os.path.join(clips_dir, row["path"]), dtype="float32")
            assert sr == 16000, f"Sample rate {sr} != 16000 for {row['path']}"
            assert audio.ndim == 1, f"Stereo audio for {row['path']}"
            assert audio.dtype == np.float32, f"Wrong dtype for {row['path']}"

    @pytest.mark.parametrize("split", ["dev", "test"])
    def test_duration_matches_tsv(self, split):
        """Actual audio duration should be within tolerance of TSV duration_ms."""
        rows = _tsv_rows(split)
        clips_dir = os.path.join(DATA_DIR, split, "clips")
        sample = rows[:5] + rows[len(rows)//2:len(rows)//2+5] + rows[-5:]
        for row in sample:
            audio, sr = sf.read(os.path.join(clips_dir, row["path"]), dtype="float32")
            actual_ms = int(len(audio) / sr * 1000)
            expected_ms = int(row["duration_ms"])
            assert abs(actual_ms - expected_ms) <= 50, (
                f"Duration mismatch for {row['path']}: TSV={expected_ms}ms, actual={actual_ms}ms"
            )

    @pytest.mark.parametrize("split", ["dev", "test"])
    def test_max_duration_under_30s(self, split):
        """All concatenated clips should be ≤ 30.5s (Whisper limit)."""
        rows = _tsv_rows(split)
        over = [r for r in rows if int(r["duration_ms"]) > 30500]
        assert not over, f"{len(over)} clips exceed 30.5s in {split}"


# ── Data consistency ─────────────────────────────────────────────────────────

class TestDataConsistency:
    @pytest.mark.parametrize("split", ["train", "dev", "test"])
    def test_no_duplicate_paths(self, split):
        """Audio file paths should be unique within a split."""
        rows = _tsv_rows(split)
        paths = [r["path"] for r in rows]
        assert len(paths) == len(set(paths)), f"Duplicate paths in {split}"

    @pytest.mark.parametrize("split", ["train", "dev", "test"])
    def test_no_orphan_audio_files(self, split):
        """All audio files in clips/ should be referenced in TSV."""
        rows = _tsv_rows(split)
        tsv_paths = {r["path"] for r in rows}
        clips_dir = os.path.join(DATA_DIR, split, "clips")
        if not os.path.isdir(clips_dir):
            pytest.skip(f"{split}/clips not found")
        disk_files = set(os.listdir(clips_dir))
        orphans = disk_files - tsv_paths
        assert not orphans, f"{len(orphans)} orphan files in {split}/clips: {list(orphans)[:5]}"

    @pytest.mark.parametrize("split", ["train", "dev", "test"])
    def test_multiline_sentences_have_multiclips(self, split):
        """Sentences with \\n should have num_clips > 1."""
        rows = _tsv_rows(split)
        for row in rows:
            if "\n" in row["sentence"]:
                assert int(row["num_clips"]) > 1, (
                    f"Multi-line sentence but num_clips=1 for {row['path']}"
                )