"""A module providing testing collection functions to assist with pancad testing."""
from __future__ import annotations

from functools import cache
import os
from pathlib import Path
import tomllib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

@cache
def read_test_data_file(path: Path) -> dict[str, Any]:
    """Returns the data from the test's toml file."""
    with open(path, "rb") as file:
        return tomllib.load(file)

@cache
def resolve_test_data_path(fixture_name: str) -> Path:
    """Returns the fixture's associated test data file path.

    :raises FileNotFoundError: When the file for the fixture name could not be found.
    """
    try:
        path = next(v for k, v in _map_data_paths().items() if fixture_name.startswith(k))
    except StopIteration as exc:
        raise FileNotFoundError(fixture_name) from exc
    return path

@cache
def _map_data_paths() -> dict[str, Path]:
    # Returns a mapping of the toml filename with no extension to the path of the datafile.
    paths: dict[str, Path] = {}
    data_path = Path(__file__).parent / ".." / "data"
    for dirpath, _, filenames in os.walk(data_path):
        dirpath_path = Path(dirpath)
        for name in filenames:
            path = dirpath_path / name
            if path.suffix == ".toml":
                paths[path.stem] = path
    return paths
