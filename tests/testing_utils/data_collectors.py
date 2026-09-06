"""A module providing testing collection functions to assist with pancad testing."""
from __future__ import annotations

from functools import cache
import math
import os
from pathlib import Path
import tomllib
from typing import TYPE_CHECKING

from pancad.constants import SketchConstraint as SC
from pancad.utils import trigonometry as trig, quat

from tests._typing import GeometrySpec, ConstraintSpec, FeatureSpec

if TYPE_CHECKING:
    from typing import Any

    from pancad.utils.pancad_types import SpaceVector

    from tests._typing import GeometrySampleData, ChangeTest, TestGroup

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

def read_quaternion(raw_quat: Any) -> quat.Quat:
    """Returns a quaternion from an unknown datatype, usually read from a toml file.

    :param raw_quat: An object that should be a dictionary with 'angle' and 'axis' keys.
    :raises AssertionError: When the axis vector's length is not 3.
    """
    angle = math.radians(float(raw_quat["angle"]))
    axis = read_vector(raw_quat["axis"])
    assert len(axis) == 3
    return quat.Quat.from_angle(angle, axis)

def read_geometry_data_entry(entry: dict[str, Any]) -> GeometrySampleData:
    """Reads a geometry entry from a data file and converts its vectors/scalars from Any to floats
    unit vectors, and radians as specified in the file.
    """
    vectors = {k: read_vector(v) for k, v in entry.get("vectors", {}).items()}
    vectors.update(
        # Normalize to a unit vector and add any normed_vectors
        {k: read_vector(v, True) for k, v in entry.get("normed_vectors", {}).items()}
    )
    vectors.update( # Convert polar_spherical_vectors degrees inputs to radians
        {k: read_vector(v, polar_spherical=True)
         for k, v in entry.get("polar_spherical_vectors", {}).items()}
    )
    scalars = {k: float(v) for k, v in entry.get("scalars", {}).items()}
    scalars.update( # Covert degree scalars to radians
        {k: math.radians(v) for k, v in entry.get("degree_scalars", {}).items()}
    )
    quats = {k: read_quaternion(v) for k, v in entry.get("quats", {}).items()}
    if vectors or scalars or quats:
        return {"vectors": vectors, "scalars": scalars, "quats": quats}
    raise LookupError("No vectors, scalars, or quaternions found")

def read_geometry_spec(name: str, spec: dict[str, Any], type_: str | None=None) -> GeometrySpec:
    """Reads a geometry's name and parameters from a data file's dictionary."""
    if not type_: # Data file specified or
        type_ = spec["type"]
    return GeometrySpec(name, type_, spec.get("construction", False),
                        read_geometry_data_entry(spec))

def read_constraint_spec(name: str, spec: dict[str, Any]) -> ConstraintSpec:
    """Reads a constraint's name and parameters from a data file's dictionary."""
    params: GeometrySampleData | None
    try:
        params = read_geometry_data_entry(spec)
    except LookupError:
        params = None # No vectors or scalars found, so this constraint doesn't need params.
    return ConstraintSpec(str(name), SC(spec["type"]), tuple(spec["geometry"]), params,
                          quadrant=spec.get("quadrant"),
                          unit=spec.get("unit"),
                          is_radians=spec.get("is_radians"))

def read_feature_spec(name: str, spec: dict[str, Any]) -> FeatureSpec:
    """Reads a feature's name and parameters from a data file's dictionary."""
    geometry = tuple(read_geometry_spec(n, s) for n, s in spec.get("geometry", {}).items())
    constraints = tuple(read_constraint_spec(n, s)
                        for n, s in spec.get("constraints", {}).items())
    pose: GeometrySpec | None = None
    if "pose" in spec:
        pose = read_geometry_spec(f"{name}_pose", spec["pose"], "pose")
    return FeatureSpec(str(name), str(spec["type"]),
                       {"geometry": geometry, "constraints": constraints, "pose": pose})

def read_feature_data(data: dict[str, Any], *keys: str) -> dict[tuple[str, ...], FeatureSpec]:
    """Reads a nested dictionary of feature settings into a named tuple.

    :raises LookupError: When one of the provided keys could not be found in the data
    """
    for key in keys:
        try:
            data = data[key]
        except KeyError as exc:
            raise LookupError(f"Could not find '{key}' in chain '{'.'.join(keys)}'") from exc
    return {keys + (k,): read_feature_spec(k, v) for k, v in data.items()}

def read_geometry_data(data: dict[str, Any],
                        *keys: str) -> dict[tuple[str, ...], GeometrySampleData]:
    """Reads a nested dictionary of geometry into a dictionary of the nested keys mapped to the
    geometry data.

    :raises LookupError: When one of the provided keys could not be found in the data or if no
        vectors or scalars are found in one of the data entries.
    """
    for key in keys:
        try:
            data = data[key]
        except KeyError as exc:
            raise LookupError(f"Could not find '{key}' in chain '{'.'.join(keys)}'") from exc
    keyed_data = {keys + (k,): v for k, v in data.items()}
    while True:
        try:
            return {k: read_geometry_data_entry(v) for k, v in keyed_data.items()}
        except LookupError:
            keyed_data ={k + (sk,): sv for k, v in keyed_data.items() for sk, sv in v.items()}

def make_geometry_sample_input(fixture_name: str) -> TestGroup[GeometrySampleData]:
    """Converts the raw data read from a fixture's sample data file into a list of test ids and a
    list of GeometrySampleData dictionaries.
    """
    raw_data = read_test_data_file(resolve_test_data_path(fixture_name))
    keys = resolve_test_data_keys(fixture_name, raw_data)
    data = read_geometry_data(raw_data, *keys)
    return [".".join(id_) for id_ in data], list(data.values())

def make_geometry_change_input(fixture_name: str) -> TestGroup[ChangeTest]:
    """Converts the raw data read from a fixture's sample data file into a list of test ids and a
    list of GeometrySampleData dictionary pairs. The first of the pair is the starting geometry
    and the second specifies the change to perform on the starting geometry.
    """
    raw_data = read_test_data_file(resolve_test_data_path(fixture_name))
    keys = resolve_test_data_keys(fixture_name, raw_data)
    data = read_geometry_data(raw_data, *keys)
    ids: list[str] = []
    tests: list[tuple[GeometrySampleData, GeometrySampleData]] = []

    for initial_key, initial_geometry in [(k, v) for k, v in data.items() if k[-1] == "initial"]:
        # Match up initial geometry with any geometry that starts with the same set of keys.
        group_ids: list[str] = []
        group_tests: list[tuple[GeometrySampleData, GeometrySampleData]] = []
        for id_, change_geometry in data.items():
            if initial_key[:-1] == id_[:-1] and initial_key != id_:
                group_ids.append(".".join(id_))
                group_tests.append((initial_geometry, change_geometry))
        if not group_ids:
            raise ValueError(f"No changes found for initial geometry: {initial_key}")
        ids.extend(group_ids)
        tests.extend(group_tests)
    return ids, tests

def make_feature_sample_input(fixture_name: str) -> TestGroup[FeatureSpec]:
    """Converts raw data read from a fixture's 'feat_' file into a list of test ids and a list of
    FeatureSpec pair.
    """
    raw_data = read_test_data_file(resolve_test_data_path(fixture_name))
    keys = resolve_test_data_keys(fixture_name, raw_data)
    data = read_feature_data(raw_data, *keys)
    return [".".join(id_) for id_ in data], list(data.values())
