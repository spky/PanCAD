"""Module defining pytest fixture configuration"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING
import sys

import pytest

from pancad.constants import ConstraintReference as CR, SketchConstraint as SC
from pancad.constraints._generator import make_constraint
from pancad.geometry.line_segment import LineSegment
from pancad.geometry.system import TwoDSketchSystem
from pancad.geometry.coordinate_system import Pose
from pancad.geometry.sketch import Sketch
from pancad.filetypes.part_file import PartFile
from pancad.constraints.state_constraint import AlignAxes
from pancad.constants import FeatureType as FT
from pancad.geometry.extrude import Extrude, ExtrudeSettings
from pancad.utils import trigonometry as trig, quat

from tests.testing_utils import sketch_gen
from tests.testing_utils.data_collectors import read_test_data_file, resolve_test_data_path
from tests._typing import GeometrySpec, ConstraintSpec, FeatureSpec

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from pancad.utils.pancad_types import SpaceVector
    from tests._typing import GeometrySampleData, ChangeTest, TestGroup

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

def read_quaternion(raw_quat: Any) -> quat.Quat:
    """Returns a quaternion from an unknown datatype, usually read from a toml file.

    :param raw_quat: An object that should be a dictionary with 'angle' and 'axis' keys.
    :raises AssertionError: When the axis vector's length is not 3.
    """
    angle = math.radians(float(raw_quat["angle"]))
    axis = read_vector(raw_quat["axis"])
    assert len(axis) == 3
    return quat.Quat.from_angle(angle, axis)

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

def _read_geometry_data_entry(entry: dict[str, Any]) -> GeometrySampleData:
    # Reads a geometry entry from a data file and converts its vectors/scalars from Any to floats
    # unit vectors, and radians as specified in the file.
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

def _read_geometry_spec(name: str, spec: dict[str, Any], type_: str | None=None) -> GeometrySpec:
    # Reads a geometry's name and parameters from a data file's dictionary.
    if not type_: # Data file specified or
        type_ = spec["type"]
    return GeometrySpec(name, type_, spec.get("construction", False),
                        _read_geometry_data_entry(spec))

def _read_constraint_spec(name: str, spec: dict[str, Any]) -> ConstraintSpec:
    # Reads a constraint's name and parameters from a data file's dictionary.
    params: GeometrySampleData | None
    try:
        params = _read_geometry_data_entry(spec)
    except LookupError:
        params = None # No vectors or scalars found, so this constraint doesn't need params.
    return ConstraintSpec(str(name), SC(spec["type"]), tuple(spec["geometry"]), params,
                          quadrant=spec.get("quadrant"),
                          unit=spec.get("unit"),
                          is_radians=spec.get("is_radians"))

def _read_feature_spec(name: str, spec: dict[str, Any]) -> FeatureSpec:
    geometry = tuple(_read_geometry_spec(n, s) for n, s in spec.get("geometry", {}).items())
    constraints = tuple(_read_constraint_spec(n, s)
                        for n, s in spec.get("constraints", {}).items())
    pose: GeometrySpec | None = None
    if "pose" in spec:
        pose = _read_geometry_spec(f"{name}_pose", spec["pose"], "pose")
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
    return {keys + (k,): _read_feature_spec(k, v) for k, v in data.items()}

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
            return {k: _read_geometry_data_entry(v) for k, v in keyed_data.items()}
        except LookupError:
            keyed_data ={k + (sk,): sv for k, v in keyed_data.items() for sk, sv in v.items()}

def _make_geometry_sample_input(fixture_name: str) -> TestGroup[GeometrySampleData]:
    """Converts the raw data read from a fixture's sample data file into a list of test ids and a
    list of GeometrySampleData dictionaries.
    """
    raw_data = read_test_data_file(resolve_test_data_path(fixture_name))
    keys = resolve_test_data_keys(fixture_name, raw_data)
    data = read_geometry_data(raw_data, *keys)
    return [".".join(id_) for id_ in data], list(data.values())

def _make_geometry_change_input(fixture_name: str) -> TestGroup[ChangeTest]:
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

def _make_feature_sample_input(fixture_name: str) -> TestGroup[FeatureSpec]:
    """Converts raw data read from a fixture's 'feat_' file into a list of test ids and a list of
    FeatureSpec pair.
    """
    raw_data = read_test_data_file(resolve_test_data_path(fixture_name))
    keys = resolve_test_data_keys(fixture_name, raw_data)
    data = read_feature_data(raw_data, *keys)
    return [".".join(id_) for id_ in data], list(data.values())

def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Generates tests from the names of the fixtures when the match patterns like data_ and
    changes_
    """
    for data_fixture in [f for f in metafunc.fixturenames if f.startswith("data_")]:
        if data_fixture == "data_regression":
            continue
        ids, sample_data = _make_geometry_sample_input(data_fixture)
        metafunc.parametrize(data_fixture, sample_data, ids=ids)

    for change_fixture in [f for f in metafunc.fixturenames if f.startswith("changes_")]:
        ids, change_data = _make_geometry_change_input(change_fixture)
        metafunc.parametrize(change_fixture, change_data, ids=ids)

    for feat_fixture in [f for f in metafunc.fixturenames if f.startswith("feat_")]:
        ids, feat_data = _make_feature_sample_input(feat_fixture)
        metafunc.parametrize(feat_fixture, feat_data, ids=ids)

@pytest.fixture(name="min_squareable")
def fixture_min_squareable() -> float:
    """Returns the minimum squareable floating point number on this system."""
    # When the square of a float is smaller than the smallest IEEE subnormal, the new float is
    # set to 0.
    info = sys.float_info
    min_squareable_exp = (info.min_exp - info.mant_dig) // 2
    squareable_limit = 1 / math.sqrt(info.radix) * info.radix ** min_squareable_exp
    min_squareable = math.nextafter(squareable_limit, 1)
    assert squareable_limit**2  == 0 # check the squareable_limit's square is actually 0
    assert min_squareable**2 != 0 # check that min_squareable's square is actually not 0
    return min_squareable

@pytest.fixture(name="min_full_precision_squareable")
def fixture_min_full_precision_squareable() -> float:
    """Returns the minimum squareable floating point number on this system that still maintains
    full precision, meaning it is not subnormal.
    """
    min_squareable = math.sqrt(sys.float_info.min)
    assert math.sqrt(min_squareable**2) == min_squareable
    return min_squareable

@pytest.fixture(name="unconstrained_square_sketch")
def fixture_unconstrained_square_sketch() -> Sketch:
    """Square sketch with just the lines, no constraints."""
    side = 1
    bottom_left = (0, 0)
    bottom_right = (side, 0)
    top_left = (0, side)
    top_right = (side, side)

    bottom = LineSegment(bottom_left, bottom_right)
    right = LineSegment(bottom_right, top_right)
    top = LineSegment(top_right, top_left)
    left = LineSegment(top_left, bottom_left)
    system = TwoDSketchSystem([bottom, right, top, left])
    pose = Pose.from_yaw_pitch_roll((0, 0, 0), 0, 0, 0)
    return Sketch(system, pose)

@pytest.fixture(name="joined_square_sketch")
def fixture_joined_square_sketch(unconstrained_square_sketch) -> Sketch:
    """Square sketch with just the lines. Line end points are coincident."""
    sketch = unconstrained_square_sketch
    bottom, right, top, left = sketch.geometry_system.geometry
    sketch.geometry_system.constraints.extend(
        [
            make_constraint(SC.COINCIDENT, bottom.start, left.end),
            make_constraint(SC.COINCIDENT, bottom.end, right.start),
            make_constraint(SC.COINCIDENT, right.end, top.start),
            make_constraint(SC.COINCIDENT, top.end, left.start),
        ]
    )
    return sketch

@pytest.fixture(name="square_sketch_bottom_length")
def fixture_square_sketch_bottom_length(joined_square_sketch) -> Sketch:
    """Square sketch with the bottom line length constrained."""
    sketch = joined_square_sketch
    unit = "mm"
    bottom, *_ = sketch.geometry_system.geometry
    side_length = bottom.end.x - bottom.start.x
    distance = make_constraint(SC.DISTANCE, bottom.start, bottom.end,
                               value=side_length, unit=unit)
    sketch.geometry_system.constraints.append(distance)
    return sketch

CSYS = -1
BOTTOM = 0
RIGHT = 1
TOP = 2
LEFT = 3

CONSTRAINT_PARAMS = [
    (
        # Bottom/Right Equal, Bottom/Top Horiz, Left/Right Vert, Bottom Start
        # *point* coincident to origin.
        (SC.HORIZONTAL, ((BOTTOM, CR.CORE),)),
        (SC.VERTICAL, ((RIGHT, CR.CORE),)),
        (SC.HORIZONTAL, ((TOP, CR.CORE),)),
        (SC.VERTICAL, ((LEFT, CR.CORE),)),
        (SC.EQUAL, ((BOTTOM, CR.CORE), (RIGHT, CR.CORE))),
        (SC.COINCIDENT, ((BOTTOM, CR.START), (CSYS, CR.ORIGIN))),
    ),
    (
        # All Equal, Bottom Horizontal, Bottom/Left Perpendicular, Bottom Start
        # *point* coincident to origin.
        (SC.HORIZONTAL, ((BOTTOM, CR.CORE),)),
        (SC.EQUAL, ((BOTTOM, CR.CORE), (RIGHT, CR.CORE))),
        (SC.EQUAL, ((BOTTOM, CR.CORE), (TOP, CR.CORE))),
        (SC.EQUAL, ((BOTTOM, CR.CORE), (LEFT, CR.CORE))),
        (SC.PERPENDICULAR, ((BOTTOM, CR.CORE), (LEFT, CR.CORE))),
        (SC.COINCIDENT, ((BOTTOM, CR.START), (CSYS, CR.ORIGIN))),
    ),
    (
        # Bottom Horizontal, Bottom/Left Perpendicular, R/L parallel, B/T
        # parallel, Bottom Start *point* coincident to origin.
        (SC.HORIZONTAL, ((BOTTOM, CR.CORE),)),
        (SC.EQUAL, ((BOTTOM, CR.CORE), (RIGHT, CR.CORE))),
        (SC.PERPENDICULAR, ((BOTTOM, CR.CORE), (LEFT, CR.CORE))),
        (SC.PARALLEL, ((RIGHT, CR.CORE), (LEFT, CR.CORE))),
        (SC.PARALLEL, ((BOTTOM, CR.CORE), (TOP, CR.CORE))),
        (SC.COINCIDENT, ((BOTTOM, CR.START), (CSYS, CR.ORIGIN))),
    ),
    (
        # Bottom/Right Equal, Bottom/Top Horiz, Left/Right Vert, Bottom Left
        # *lines* coincident to origin.
        (SC.HORIZONTAL, ((BOTTOM, CR.CORE),)),
        (SC.VERTICAL, ((RIGHT, CR.CORE),)),
        (SC.HORIZONTAL, ((TOP, CR.CORE),)),
        (SC.VERTICAL, ((LEFT, CR.CORE),)),
        (SC.EQUAL, ((BOTTOM, CR.CORE), (RIGHT, CR.CORE))),
        (SC.COINCIDENT, ((BOTTOM, CR.CORE), (CSYS, CR.ORIGIN))),
        (SC.COINCIDENT, ((LEFT, CR.CORE), (CSYS, CR.ORIGIN))),
    ),
    (
        # All Sides equal, bottom horizontal, left vertical, bottom left
        # coincident to origin
        (SC.HORIZONTAL, ((BOTTOM, CR.CORE),)),
        (SC.VERTICAL, ((LEFT, CR.CORE),)),
        (SC.EQUAL, ((BOTTOM, CR.CORE), (RIGHT, CR.CORE))),
        (SC.EQUAL, ((BOTTOM, CR.CORE), (TOP, CR.CORE))),
        (SC.EQUAL, ((BOTTOM, CR.CORE), (LEFT, CR.CORE))),
        (SC.COINCIDENT, ((BOTTOM, CR.START), (CSYS, CR.ORIGIN))),
    ),

]
@pytest.fixture(name="square_sketch_variations", params=CONSTRAINT_PARAMS)
def fixture_square_sketch_variations(request, square_sketch_bottom_length):
    """Variations on a fully constrained square sketch. Same geometry, varied
    constraints.
    """
    sketch = square_sketch_bottom_length
    constraints = []
    for type_, refs in request.param:
        input_geometry = []
        for index, constraint_ref in refs:
            geo = sketch.geometry_system.geometry[index]
            input_geometry.append(geo.get_reference(constraint_ref))
        constraints.append(make_constraint(type_, *input_geometry))
    sketch.geometry_system.constraints.extend(constraints)
    yield sketch

@pytest.fixture(name="line_angled_to_x_axis_sketches")
def fixture_line_angled_to_x_axis_sketches() -> list[Sketch]:
    """A list of angle-sweeping sketches placing a single line segment in
    different quadrants relative to the sketch's x-axis. Useful for checking the
    implementation of angle constraints inside a single CAD file.
    """
    angle_sweep_params = [
        # Quadrant, Angle (Degrees), start_to_end
        (1, 45, False),
        (2, 45, False),
        (3, 45, False),
        (4, 45, False),
        (1, 45, True),
        (2, 45, True),
        (3, 45, True),
        (4, 45, True),
    ]
    sketches = []
    for quadrant, angle, start_radially_out in angle_sweep_params:
        sketch = sketch_gen.sketch_with_line_angled_to_x_axis(quadrant, angle, start_radially_out)
        sketch.name = f"Quadrant_{quadrant}_Angle_{angle}"
        if start_radially_out:
            sketch.name = sketch.name + "_EndOnOrigin"
        else:
            sketch.name = sketch.name + "_StartOnOrigin"
        sketches.append(sketch)
    return sketches


@pytest.fixture(name="empty_part_file")
def fixture_empty_part_file() -> PartFile:
    """A partfile with nothing in it."""
    return PartFile("EmptyTestPart")

@pytest.fixture(name="square_sketch_part_file")
def fixture_square_sketch_part_file() -> PartFile:
    """A partfile with just a square sketch inside it"""
    part = PartFile("SquareSketchPartTest")
    sketch = sketch_gen.square()
    constraints = [
        AlignAxes(part.container.feature_system.coordinate_system,
                  sketch.pose.coordinate_system),
    ]
    part.container.feature_system.features.append(sketch)
    part.container.feature_system.constraints.extend(constraints)
    return part

@pytest.fixture(name="cube_part_file")
def fixture_cube_part_file() -> PartFile:
    """A partfile with just a square sketch and extrude inside it"""
    part = PartFile("CubePartTest")
    sketch = sketch_gen.square()
    extrude_settings = ExtrudeSettings(type_=FT.DIMENSION, length=1, unit="mm")
    extrude = Extrude(sketch, extrude_settings, name="CubeExtrude")
    constraints = [
        AlignAxes(part.container.feature_system.coordinate_system,
                  sketch.pose.coordinate_system),
    ]
    part.container.feature_system.features.append(sketch)
    part.container.feature_system.constraints.extend(constraints)
    part.container.feature_system.features.append(extrude)
    return part

@pytest.fixture(name="cylinder_part_file")
def fixture_cylinder_part_file() -> PartFile:
    """A PartFile with a cylinder sketch and extrusion inside it."""
    part = PartFile("CylinderPartTest")
    sketch = sketch_gen.circle()
    extrude_settings = ExtrudeSettings(type_=FT.DIMENSION, length=1, unit="mm")
    extrude = Extrude(sketch, extrude_settings, name="CylinderExtrude")
    constraints = [
        AlignAxes(part.container.feature_system.coordinate_system,
                  sketch.pose.coordinate_system),
    ]
    part.container.feature_system.features.append(sketch)
    part.container.feature_system.constraints.extend(constraints)
    part.container.feature_system.features.append(extrude)
    return part

@pytest.fixture(name="rounded_edge_cube_part_file")
def fixture_rounded_edge_cube_part_file() -> PartFile:
    """A PartFile with a rounded square sketch and extrusion inside it."""
    part = PartFile("RoundedEdgeCubePartTest")
    sketch = sketch_gen.rounded_square()
    extrude_settings = ExtrudeSettings(type_=FT.DIMENSION, length=1, unit="mm")
    extrude = Extrude(sketch, extrude_settings, name="RoundedSquareExtrude")
    constraints = [
        AlignAxes(part.container.feature_system.coordinate_system,
                  sketch.pose.coordinate_system),
    ]
    part.container.feature_system.features.append(sketch)
    part.container.feature_system.constraints.extend(constraints)
    part.container.feature_system.features.append(extrude)
    return part

@pytest.fixture(name="ellipse_part_file")
def fixture_ellipse_part_file() -> PartFile:
    """A partfile with just a a single ellipse sketch and extrusion."""
    part = PartFile("EllipseExtrudePartTest")
    sketch = sketch_gen.ellipse()
    extrude_settings = ExtrudeSettings(type_=FT.DIMENSION, length=1, unit="mm")
    extrude = Extrude(sketch, extrude_settings, name="EllipseExtrude")
    constraints = [
        AlignAxes(part.container.feature_system.coordinate_system,
                  sketch.pose.coordinate_system),
    ]
    part.container.feature_system.features.append(sketch)
    part.container.feature_system.constraints.extend(constraints)
    part.container.feature_system.features.append(extrude)
    return part

@pytest.fixture(name="square_variations_part_file")
def fixture_square_variations_part_file(square_sketch_variations) -> PartFile:
    """A PartFile for each square_variations sketch the sketch and an extrusion inside it."""
    sketch = square_sketch_variations
    part = PartFile("square_sketch_variations")
    extrude_settings = ExtrudeSettings(type_=FT.DIMENSION, length=1, unit="mm")
    extrude = Extrude(sketch, extrude_settings, name="SquareVariationExtrude")
    constraints = [
        AlignAxes(part.container.feature_system.coordinate_system,
                  sketch.pose.coordinate_system),
    ]
    part.container.feature_system.features.append(sketch)
    part.container.feature_system.constraints.extend(constraints)
    part.container.feature_system.features.append(extrude)
    return part

@pytest.fixture(name="angle_dimension_sweep_part_file")
def fixture_angle_dimension_sweep_part_file(line_angled_to_x_axis_sketches) -> PartFile:
    """A PartFile with a sketch containing a sweep of angles for checking angle translation."""
    part = PartFile("angle_dimension_sweep")
    for sketch in line_angled_to_x_axis_sketches:
        part.container.feature_system.features.append(sketch)
        part.container.feature_system.constraints.append(
            AlignAxes(part.container.feature_system.coordinate_system,
                      sketch.pose.coordinate_system)
        )
    return part
