"""Tests for the segment synthesis cache (cache.py)."""

import pytest

from video_dubber.audio import load_wav, make_sine
from video_dubber.cache import (
    CachedVoiceCloner,
    CacheError,
    SynthesisCache,
    cache_key,
    default_cache_dir,
)
from video_dubber.cli import main
from video_dubber.voice import ScriptedVoiceCloner, SpeakerProfile


def _profile(name: str = "asha", pitch: float = 180.0, wpm: float = 150.0) -> SpeakerProfile:
    return SpeakerProfile(
        name=name,
        embedding=[0.25] * 16,
        base_pitch_hz=pitch,
        speaking_rate_wpm=wpm,
    )


class _CountingCloner:
    """Wraps the scripted cloner and counts how often synthesis runs."""

    def __init__(self) -> None:
        self.inner = ScriptedVoiceCloner()
        self.calls = 0

    @property
    def name(self) -> str:
        return self.inner.name

    def clone(self, name, audios, reference_files=None):
        return self.inner.clone(name, audios, reference_files)

    def synthesize(self, text, profile, words_per_minute=None):
        self.calls += 1
        return self.inner.synthesize(text, profile, words_per_minute)


# --- cache_key ---------------------------------------------------------------


def test_key_is_deterministic():
    profile = _profile()
    assert cache_key("scripted", "hello world", profile) == cache_key(
        "scripted", "hello world", profile
    )


def test_key_changes_with_text_profile_rate_cloner_and_sample_rate():
    profile = _profile()
    base = cache_key("scripted", "hello world", profile)
    assert cache_key("scripted", "hello mars", profile) != base
    assert cache_key("scripted", "hello world", _profile(name="ravi")) != base
    assert cache_key("scripted", "hello world", _profile(pitch=220.0)) != base
    assert cache_key("scripted", "hello world", profile, words_per_minute=120.0) != base
    assert cache_key("other", "hello world", profile) != base
    assert cache_key("scripted", "hello world", profile, sample_rate=8000) != base


def test_key_ignores_embedding_float_noise():
    profile = _profile()
    noisy = SpeakerProfile(
        name=profile.name,
        embedding=[v + 1e-7 for v in profile.embedding],
        base_pitch_hz=profile.base_pitch_hz,
        speaking_rate_wpm=profile.speaking_rate_wpm,
    )
    assert cache_key("scripted", "hello world", noisy) == cache_key(
        "scripted", "hello world", profile
    )


def test_key_rejects_non_positive_wpm_or_sample_rate():
    profile = _profile()
    with pytest.raises(CacheError):
        cache_key("scripted", "hi", profile, words_per_minute=0)
    with pytest.raises(CacheError):
        cache_key("scripted", "hi", profile, sample_rate=0)


# --- SynthesisCache ----------------------------------------------------------


def test_put_get_round_trip_and_stats(tmp_path):
    cache = SynthesisCache(tmp_path / "cache")
    audio = make_sine(440.0, 0.5)
    key = cache_key("scripted", "hello world", _profile())
    assert cache.get(key) is None
    cache.put(key, audio)
    got = cache.get(key)
    assert got is not None
    assert got.sample_rate == audio.sample_rate
    assert got.samples == pytest.approx(audio.samples, abs=1 / 32767 + 1e-9)
    stats = cache.stats()
    assert stats.entries == 1
    assert stats.total_bytes > 0
    assert stats.hits == 1
    assert stats.misses == 1
    assert stats.hit_rate == pytest.approx(0.5)


def test_index_survives_reopen(tmp_path):
    cache = SynthesisCache(tmp_path / "cache")
    key = cache_key("scripted", "hello world", _profile())
    cache.put(key, make_sine(440.0, 0.3))
    reopened = SynthesisCache(tmp_path / "cache")
    assert reopened.stats().entries == 1
    assert reopened.get(key) is not None
    assert reopened.stats().hits == 1


def test_corrupt_index_starts_empty(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "index.json").write_text("{not json", encoding="utf-8")
    cache = SynthesisCache(cache_dir)
    assert cache.stats().entries == 0
    key = cache_key("scripted", "hello world", _profile())
    cache.put(key, make_sine(440.0, 0.3))
    assert cache.get(key) is not None


def test_missing_entry_file_counts_as_miss(tmp_path):
    cache = SynthesisCache(tmp_path / "cache")
    key = cache_key("scripted", "hello world", _profile())
    cache.put(key, make_sine(440.0, 0.3))
    (tmp_path / "cache" / f"{key}.wav").unlink()
    assert cache.get(key) is None
    assert cache.stats().entries == 0


def test_unreadable_entry_file_counts_as_miss(tmp_path):
    cache = SynthesisCache(tmp_path / "cache")
    key = cache_key("scripted", "hello world", _profile())
    cache.put(key, make_sine(440.0, 0.3))
    (tmp_path / "cache" / f"{key}.wav").write_bytes(b"not a wav")
    assert cache.get(key) is None
    assert cache.stats().entries == 0


def test_prune_evicts_least_recently_used(tmp_path):
    # Budgets hold two entries (each ~16 KB WAV) but not three.
    cache = SynthesisCache(tmp_path / "cache", max_bytes=40_000, max_entries=2)
    keys = [cache_key("scripted", f"line {i}", _profile()) for i in range(3)]
    for key in keys:
        cache.put(key, make_sine(440.0, 0.5))
    # The first key was least recently used; the newest two survive.
    assert cache.get(keys[0]) is None
    assert cache.get(keys[1]) is not None
    assert cache.get(keys[2]) is not None
    assert cache.stats().entries == 2


def test_clear_removes_files_and_resets_stats(tmp_path):
    cache = SynthesisCache(tmp_path / "cache")
    for i in range(2):
        cache.put(cache_key("scripted", f"line {i}", _profile()), make_sine(440.0, 0.3))
    assert cache.clear() == 2
    assert cache.stats().entries == 0
    assert cache.stats().total_bytes == 0
    assert list((tmp_path / "cache").glob("*.wav")) == []
    assert not (tmp_path / "cache" / "index.json").exists()


def test_cache_rejects_bad_budgets(tmp_path):
    with pytest.raises(CacheError):
        SynthesisCache(tmp_path / "cache", max_bytes=0)
    with pytest.raises(CacheError):
        SynthesisCache(tmp_path / "cache", max_entries=0)


def test_default_cache_dir_is_under_home_cache():
    assert str(default_cache_dir()).startswith(str(default_cache_dir().parent.parent))
    assert default_cache_dir().name == "synth"


# --- CachedVoiceCloner -------------------------------------------------------


def test_cached_cloner_synthesises_once_per_unique_request(tmp_path):
    inner = _CountingCloner()
    cached = CachedVoiceCloner(inner, SynthesisCache(tmp_path / "cache"))
    profile = _profile()
    first = cached.synthesize("hello world", profile)
    second = cached.synthesize("hello world", profile)
    assert inner.calls == 1
    assert cached.hits == 1
    assert cached.misses == 1
    assert second.samples == pytest.approx(first.samples, abs=1 / 32767 + 1e-9)
    # A different line, voice or rate is a new synthesis.
    cached.synthesize("hello mars", profile)
    cached.synthesize("hello world", _profile(name="ravi"))
    cached.synthesize("hello world", profile, words_per_minute=120.0)
    assert inner.calls == 4


def test_cached_cloner_name_and_clone_delegate(tmp_path):
    inner = _CountingCloner()
    cached = CachedVoiceCloner(inner, SynthesisCache(tmp_path / "cache"))
    assert cached.name == "scripted+cache"
    audio = make_sine(440.0, 1.0)
    profile = cached.clone("asha", [audio])
    assert profile.name == "asha"


def test_cached_cloner_survives_a_broken_cache(tmp_path, monkeypatch):
    inner = _CountingCloner()
    cache = SynthesisCache(tmp_path / "cache")
    cached = CachedVoiceCloner(inner, cache)
    monkeypatch.setattr(
        cache, "put", lambda key, audio: (_ for _ in ()).throw(OSError("disk full"))
    )
    audio = cached.synthesize("hello world", _profile())
    assert audio.duration > 0
    assert inner.calls == 1


# --- CLI ---------------------------------------------------------------------


def _write_profile(path) -> None:
    path.write_text(_profile().to_json(), encoding="utf-8")


def test_cli_synthesize_uses_cache_dir(tmp_path, capsys):
    profile_path = tmp_path / "asha.json"
    _write_profile(profile_path)
    out_a = tmp_path / "a.wav"
    out_b = tmp_path / "b.wav"
    cache_dir = tmp_path / "cache"
    assert (
        main(
            [
                "synthesize",
                "hello world",
                "--profile",
                str(profile_path),
                "--out",
                str(out_a),
                "--cache-dir",
                str(cache_dir),
            ]
        )
        == 0
    )
    first = capsys.readouterr().out
    assert "cache miss" in first
    assert (
        main(
            [
                "synthesize",
                "hello world",
                "--profile",
                str(profile_path),
                "--out",
                str(out_b),
                "--cache-dir",
                str(cache_dir),
            ]
        )
        == 0
    )
    second = capsys.readouterr().out
    assert "cache hit" in second
    assert load_wav(str(out_a)).samples == pytest.approx(
        load_wav(str(out_b)).samples, abs=1 / 32767 + 1e-9
    )


def test_cli_cache_stats_and_clear(tmp_path, capsys):
    cache_dir = tmp_path / "cache"
    cache = SynthesisCache(cache_dir)
    cache.put(cache_key("scripted", "hello world", _profile()), make_sine(440.0, 0.3))
    assert main(["cache", "--cache-dir", str(cache_dir)]) == 0
    out = capsys.readouterr().out
    assert "entries: 1" in out
    assert "hit rate: 0.00" in out
    assert main(["cache", "--cache-dir", str(cache_dir), "--clear"]) == 0
    cleared = capsys.readouterr().out
    assert "cleared 1 cached segment" in cleared
    assert SynthesisCache(cache_dir).stats().entries == 0


def test_cli_cache_stats_empty_dir(tmp_path, capsys):
    assert main(["cache", "--cache-dir", str(tmp_path / "fresh")]) == 0
    assert "entries: 0" in capsys.readouterr().out
