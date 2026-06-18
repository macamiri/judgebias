"""Offline tests for the shipped demo entry point.

No network and no ``[examples]`` extra required — the missing-extra path is
forced so these run in plain CI."""
import builtins

import judgebias.demo as demo


def test_demo_main_is_callable():
    assert callable(demo.main)


def test_missing_datasets_message_points_to_extra():
    msg = demo._missing_datasets_message()
    assert "judgebias[examples]" in msg
    assert "datasets" in msg


def test_main_handles_missing_datasets(monkeypatch, capsys):
    # Simulate the [examples] extra not being installed: make `import datasets` fail,
    # and assert main() degrades gracefully (prints the hint, does not raise).
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "datasets":
            raise ImportError("No module named 'datasets'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    demo.main()  # must not raise
    out = capsys.readouterr().out
    assert "judgebias[examples]" in out
