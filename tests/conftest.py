"""Module defining pytest fixture configuration"""
from __future__ import annotations

import math
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

from tests.testing_utils import sketch_gen
from tests.testing_utils.data_collectors import (
    make_geometry_sample_input, make_geometry_change_input, make_feature_sample_input
)

def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Generates tests from the names of the fixtures when the match patterns like data_ and
    changes_
    """
    for data_fixture in [f for f in metafunc.fixturenames if f.startswith("data_")]:
        if data_fixture == "data_regression":
            continue
        ids, sample_data = make_geometry_sample_input(data_fixture)
        metafunc.parametrize(data_fixture, sample_data, ids=ids)

    for change_fixture in [f for f in metafunc.fixturenames if f.startswith("changes_")]:
        ids, change_data = make_geometry_change_input(change_fixture)
        metafunc.parametrize(change_fixture, change_data, ids=ids)

    for feat_fixture in [f for f in metafunc.fixturenames if f.startswith("feat_")]:
        ids, feat_data = make_feature_sample_input(feat_fixture)
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
