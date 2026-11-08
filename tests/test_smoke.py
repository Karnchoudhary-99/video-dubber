from video_dubber import __version__


def test_version_is_set():
    assert __version__ == "0.1.0"


def test_cli_exits_zero_with_help(capsys):
    from video_dubber.cli import main

    assert main([]) == 0
    assert "video-dubber" in capsys.readouterr().out
