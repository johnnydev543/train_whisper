"""Tests for concat_audio.py — unit tests for grouping/concatenation logic."""

import numpy as np
import pytest

from concat_audio import group_clips_by_duration, concat_clips, load_tsv, load_durations


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_clips():
    """Short clips with durations in ms."""
    return [
        {"path": "a.mp3", "sentence": "你好", "duration_ms": 3000},
        {"path": "b.mp3", "sentence": "世界", "duration_ms": 4000},
        {"path": "c.mp3", "sentence": "測試", "duration_ms": 5000},
        {"path": "d.mp3", "sentence": "再見", "duration_ms": 6000},
        {"path": "e.mp3", "sentence": "謝謝", "duration_ms": 2000},
    ]


@pytest.fixture
def durations_map():
    return {
        "a.mp3": 3000,
        "b.mp3": 4000,
        "c.mp3": 5000,
        "d.mp3": 6000,
        "e.mp3": 2000,
    }


# ── group_clips_by_duration ─────────────────────────────────────────────────

class TestGroupClipsByDuration:
    def test_basic_grouping(self, sample_clips, durations_map):
        """Clips should be grouped so each group <= max_duration_ms."""
        groups = group_clips_by_duration(sample_clips, durations_map, target_duration_ms=10000, max_duration_ms=15000)
        for g in groups:
            total = sum(durations_map[c["path"]] for c in g)
            silence = max(0, len(g) - 1) * 500
            assert total + silence <= 15000

    def test_oversized_clip_alone(self, durations_map):
        """A clip exceeding max_duration should be placed alone."""
        clips = [{"path": "huge.mp3", "sentence": "長", "duration_ms": 35000}]
        durations = {"huge.mp3": 35000}
        groups = group_clips_by_duration(clips, durations, target_duration_ms=28000, max_duration_ms=30000)
        assert len(groups) == 1
        assert len(groups[0]) == 1
        assert groups[0][0]["path"] == "huge.mp3"

    def test_all_clips_preserved(self, sample_clips, durations_map):
        """No clips should be lost during grouping."""
        groups = group_clips_by_duration(sample_clips, durations_map, target_duration_ms=28000)
        total_clips = sum(len(g) for g in groups)
        assert total_clips == len(sample_clips)

    def test_clip_order_preserved(self, sample_clips, durations_map):
        """Clips within groups should maintain original order."""
        groups = group_clips_by_duration(sample_clips, durations_map, target_duration_ms=28000)
        flat = [c for g in groups for c in g]
        paths = [c["path"] for c in flat]
        expected = [c["path"] for c in sample_clips]
        assert paths == expected

    def test_single_clip(self, durations_map):
        """Single clip should form a single group."""
        clips = [{"path": "a.mp3", "sentence": "獨", "duration_ms": 3000}]
        durations = {"a.mp3": 3000}
        groups = group_clips_by_duration(clips, durations, target_duration_ms=28000)
        assert len(groups) == 1
        assert len(groups[0]) == 1

    def test_empty_input(self):
        """Empty input should return no groups."""
        groups = group_clips_by_duration([], {}, target_duration_ms=28000)
        assert groups == []

    def test_silence_gap_accounted(self):
        """Silence gaps should contribute to group duration limits."""
        clips = [
            {"path": "a.mp3", "sentence": "A", "duration_ms": 7000},
            {"path": "b.mp3", "sentence": "B", "duration_ms": 8000},
        ]
        durations = {"a.mp3": 7000, "b.mp3": 8000}
        # max=14500: 7000 + 500(silence) + 8000 = 15500 > 14500 → split into 2 groups
        groups = group_clips_by_duration(clips, durations, target_duration_ms=14000, max_duration_ms=14500)
        assert len(groups) == 2


# ── concat_clips ────────────────────────────────────────────────────────────

class TestConcatClips:
    def _write_wav(self, tmp_path, name, audio, sr=16000):
        """Helper to write a WAV file for testing."""
        import soundfile as sf
        path = tmp_path / name
        sf.write(str(path), audio, sr)
        return str(path)

    def test_silence_gap_inserted(self, tmp_path):
        """Silence gap of correct length should be inserted between clips."""
        sr = 16000
        clip1 = np.ones(sr, dtype=np.float32)
        clip2 = np.ones(sr, dtype=np.float32) * 2
        self._write_wav(tmp_path, "a.wav", clip1, sr)
        self._write_wav(tmp_path, "b.wav", clip2, sr)

        groups = [[
            {"path": "a.wav", "sentence": "A", "duration_ms": 1000},
            {"path": "b.wav", "sentence": "B", "duration_ms": 1000},
        ]]

        results = concat_clips(groups, str(tmp_path), target_duration_ms=28000, silence_gap_ms=500, sample_rate=sr)
        assert len(results) == 1

        audio = results[0]["audio_array"]
        # 1s + 0.5s silence + 1s = 2.5s → 40000 samples
        expected_len = sr * 2 + 8000
        assert len(audio) == expected_len

        # Check silence region is zeros
        silence_region = audio[sr: sr + 8000]
        assert np.allclose(silence_region, 0.0)

    def test_single_clip_no_silence(self, tmp_path):
        """Single clip should have no silence gap appended."""
        sr = 16000
        clip = np.ones(sr, dtype=np.float32)
        self._write_wav(tmp_path, "a.wav", clip, sr)

        groups = [[{"path": "a.wav", "sentence": "A", "duration_ms": 1000}]]
        results = concat_clips(groups, str(tmp_path), target_duration_ms=28000, sample_rate=sr)
        assert len(results) == 1
        assert len(results[0]["audio_array"]) == sr

    def test_sentence_join_with_newline(self, tmp_path):
        """Sentences should be joined with newline."""
        sr = 16000
        self._write_wav(tmp_path, "a.wav", np.zeros(sr, dtype=np.float32), sr)
        self._write_wav(tmp_path, "b.wav", np.zeros(sr, dtype=np.float32), sr)

        groups = [[
            {"path": "a.wav", "sentence": "你好", "duration_ms": 1000},
            {"path": "b.wav", "sentence": "世界", "duration_ms": 1000},
        ]]
        results = concat_clips(groups, str(tmp_path), target_duration_ms=28000, sample_rate=sr)
        assert results[0]["sentence"] == "你好\n世界"

    def test_num_clips_count(self, tmp_path):
        """num_clips should reflect actual number of source clips."""
        sr = 16000
        for i in range(3):
            self._write_wav(tmp_path, f"c{i}.wav", np.zeros(sr, dtype=np.float32), sr)

        groups = [[{"path": f"c{i}.wav", "sentence": str(i), "duration_ms": 1000} for i in range(3)]]
        results = concat_clips(groups, str(tmp_path), target_duration_ms=28000, sample_rate=sr)
        assert results[0]["num_clips"] == 3

    def test_empty_group(self, tmp_path):
        """Empty group list should return empty list."""
        results = concat_clips([], str(tmp_path), target_duration_ms=28000)
        assert results == []

    def test_duration_ms_includes_silence(self, tmp_path):
        """duration_ms should include silence gaps between clips."""
        sr = 16000
        self._write_wav(tmp_path, "a.wav", np.zeros(sr, dtype=np.float32), sr)
        self._write_wav(tmp_path, "b.wav", np.zeros(sr, dtype=np.float32), sr)

        groups = [[
            {"path": "a.wav", "sentence": "A", "duration_ms": 1000},
            {"path": "b.wav", "sentence": "B", "duration_ms": 1000},
        ]]
        results = concat_clips(groups, str(tmp_path), target_duration_ms=28000, silence_gap_ms=500, sample_rate=sr)
        assert results[0]["duration_ms"] == 2500


# ── load_tsv / load_durations ────────────────────────────────────────────────

class TestLoadTSV:
    def test_load_tsv(self, tmp_path):
        """load_tsv should parse tab-separated files correctly."""
        tsv = tmp_path / "test.tsv"
        tsv.write_text("path\tsentence\tduration_ms\na.wav\t你好\t1000\n", encoding="utf-8")
        header, rows = load_tsv(str(tsv))
        assert header == ["path", "sentence", "duration_ms"]
        assert len(rows) == 1
        assert rows[0]["path"] == "a.wav"
        assert rows[0]["sentence"] == "你好"

    def test_load_durations(self, tmp_path):
        """load_durations should parse clip_durations.tsv."""
        tsv = tmp_path / "clip_durations.tsv"
        tsv.write_text("clip\tduration\na.mp3\t3000\nb.mp3\t5000\n", encoding="utf-8")
        durs = load_durations(str(tsv))
        assert durs == {"a.mp3": 3000, "b.mp3": 5000}