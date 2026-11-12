"""Tests for the dubber.toml config file: loading, search, validation."""

from pathlib import Path

import pytest

import video_dubber.config as config_module
from video_dubber.config import ConfigError, DubberConfig, find_config, load_config


def _write_toml(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


FULL_TOML = """\
[dubber]
target_language = "hi"
source_language = "en"
backend = "argos"
voice_library = "voices.json"
bed_level = 0.3
style = "documentary"
lexicon = "lexicon.json"

[dubber.enhance]
target = -18.0
ceiling = -2.0
high_pass = 100.0
gate = -50.0
"""


class TestLoadConfig:
    def test_missing_explicit_path_is_an_error(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "nope.toml")

    def test_no_file_anywhere_gives_defaults(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = load_config()
        assert config == DubberConfig()
        assert config.path is None

    def test_full_file_parses(self, tmp_path):
        path = _write_toml(tmp_path / "dubber.toml", FULL_TOML)
        config = load_config(path)
        assert config.target_language == "hi"
        assert config.source_language == "en"
        assert config.backend == "argos"
        assert config.voice_library == "voices.json"
        assert config.bed_level == pytest.approx(0.3)
        assert config.style == "documentary"
        assert config.lexicon == "lexicon.json"
        assert config.enhance.target == pytest.approx(-18.0)
        assert config.enhance.ceiling == pytest.approx(-2.0)
        assert config.enhance.high_pass == pytest.approx(100.0)
        assert config.enhance.gate == pytest.approx(-50.0)
        assert config.path == path

    def test_partial_file_keeps_other_defaults(self, tmp_path):
        path = _write_toml(tmp_path / "dubber.toml", '[dubber]\ntarget_language = "es"\n')
        config = load_config(path)
        assert config.target_language == "es"
        assert config.backend == "dictionary"
        assert config.enhance.target == pytest.approx(-16.0)
        assert config.bed_level is None

    def test_invalid_toml_is_an_error(self, tmp_path):
        path = _write_toml(tmp_path / "dubber.toml", "[dubber\ntarget_language = ")
        with pytest.raises(ConfigError, match="invalid TOML"):
            load_config(path)

    def test_top_level_without_dubber_table_is_an_error(self, tmp_path):
        path = _write_toml(tmp_path / "dubber.toml", 'target_language = "hi"\n')
        with pytest.raises(ConfigError, match="only a \\[dubber\\] table"):
            load_config(path)

    def test_to_dict_round_trip(self, tmp_path):
        path = _write_toml(tmp_path / "dubber.toml", FULL_TOML)
        data = load_config(path).to_dict()
        assert data["target_language"] == "hi"
        assert data["style"] == "documentary"
        assert data["enhance"]["target"] == pytest.approx(-18.0)
        assert data["path"] == str(path)


class TestValidation:
    @pytest.mark.parametrize(
        "toml, message",
        [
            ('[dubber]\ntypo_key = "hi"\n', "unknown \\[dubber\\] key"),
            ('[dubber]\nbackend = "whisper"\n', "backend must be"),
            ("[dubber]\nbed_level = 1.5\n", "bed_level must be between 0 and 1"),
            ('[dubber]\nbed_level = "loud"\n', "bed_level must be a number"),
            ('[dubber]\nstyle = "shouty"\n', "invalid style in config file"),
            ("[dubber]\nstyle = 7\n", "must be a non-empty string"),
            ('[dubber]\ntarget_language = ""\n', "must be a non-empty string"),
            ("[dubber]\ntarget_language = 7\n", "must be a non-empty string"),
            (
                "[dubber.enhance]\ntarget = -10.0\nceiling = -12.0\n",
                "ceiling .* must be >= .*target",
            ),
            (
                "[dubber.enhance]\nunknown = 1.0\n",
                "unknown \\[dubber.enhance\\] key",
            ),
            ('[dubber.enhance]\ntarget = "loud"\n', "must be a number"),
        ],
    )
    def test_invalid_values_raise(self, tmp_path, toml, message):
        path = _write_toml(tmp_path / "dubber.toml", toml)
        with pytest.raises(ConfigError, match=message):
            load_config(path)

    def test_bed_level_bool_is_not_a_number(self, tmp_path):
        path = _write_toml(tmp_path / "dubber.toml", "[dubber]\nbed_level = true\n")
        with pytest.raises(ConfigError, match="must be a number"):
            load_config(path)


class TestFindConfig:
    def test_explicit_path(self, tmp_path):
        path = _write_toml(tmp_path / "custom.toml", "[dubber]\n")
        assert find_config(path) == path
        assert find_config(tmp_path / "missing.toml") is None

    def test_cwd_dubber_toml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert find_config() is None
        path = _write_toml(tmp_path / "dubber.toml", "[dubber]\n")
        assert find_config() == path

    def test_user_config_dir(self, tmp_path, monkeypatch):
        user_dir = tmp_path / "usercfg"
        user_dir.mkdir()
        path = _write_toml(user_dir / "dubber.toml", "[dubber]\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(config_module, "_USER_CONFIG_DIR", path)
        assert find_config() == path

    def test_cwd_beats_user_config(self, tmp_path, monkeypatch):
        user_dir = tmp_path / "usercfg"
        user_dir.mkdir()
        user_path = _write_toml(user_dir / "dubber.toml", "[dubber]\n")
        local_path = _write_toml(tmp_path / "dubber.toml", "[dubber]\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(config_module, "_USER_CONFIG_DIR", user_path)
        assert find_config() == local_path
