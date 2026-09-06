"""A module providing testing collection functions to assist with pancad testing."""
from __future__ import annotations

from functools import cache
import math
import os
from pathlib import Path
import tomllib
from typing import TYPE_CHECKING

from pancad.utils import trigonometry as trig

if TYPE_CHECKING:
    from typing import Any

    from pancad.utils.pancad_types import SpaceVector

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

def resolve_test_data_keys(fixture_name: str, data: dict[str, Any]) -> list[str]:
    """Returns a list of keys found in the fixture's name that match keys in the data.
    Progressively searches down the data's nested dictionary for keys that match the start of the
    fixture name with the previous names removed. Underscores preceding keys are ignored.

    :raises LookupError: When a key cannot be found.
    :raises RuntimeError: When the string loop fails to reduce the length of the key string.
    """
    key_str = fixture_name.removeprefix(resolve_test_data_path(fixture_name).stem).lstrip("_")
    keys: list[str] = []
    while key_str:
        check_str = key_str
        try:
            key, data = next((k, v) for k, v in data.items() if key_str.startswith(k))
        except StopIteration as exc:
            raise LookupError("Could not find a key with the same start as"
                              f"{key_str} in {data.keys()}") from exc
        keys.append(key)
        key_str = key_str.removeprefix(key).lstrip("_")
        if check_str == key_str:
            raise RuntimeError("Loop stuck, check/key strings match even after prefix removed."
                               f" key string: {key_str}")
    return keys

def read_vector(raw_vector: Any,
                normalize: bool=False,
                polar_spherical: bool=False) -> SpaceVector:
    """Returns a 2 or 3 float long vector from an unknown datatype, usually read from a toml file.

    :param raw_vector: An object that should be a vector.
    :param normalize: Whether to normalize the vector before returning it.
    :param polar_spherical: Whether the 2nd and 3rd (if present) components should be converted to
        radians.

    :raises ValueError: When the vector components could not be converted into floats.
    :raises AssertionError: When the vector's length is not 2 or 3.
    """
    vector = tuple(map(float, raw_vector))
    assert len(vector) == 2 or len(vector) == 3
    if polar_spherical:
        if len(vector) == 2: # Polar
            vector = (vector[0], math.radians(vector[1]))
        else: # Spherical
            vector = (vector[0], math.radians(vector[1]), math.radians(vector[2]))
    elif normalize:
        vector = trig.to_1d_tuple(trig.get_unit_vector(vector))
    return vector

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
