from __future__ import annotations

from ml.cli.support import run_cli
from ml.errors import SophiaUsageError


def test_run_cli_propagates_integer_return_codes() -> None:
    assert run_cli(lambda: 0) == 0
    assert run_cli(lambda: 7) == 7
    assert run_cli(lambda: None) == 0


def test_run_cli_maps_usage_errors_to_two(capsys) -> None:
    def fail() -> None:
        raise SophiaUsageError("bad input")

    assert run_cli(fail) == 2
    assert "bad input" in capsys.readouterr().err
