"""Tests for methods shared by all pancad geometry (and features when applicable).
"""

import pytest

from pancad.api import(
    Axis,
    Circle,
    CircularArc,
    CoordinateSystem,
    Ellipse,
    Line,
    LineSegment,
    Plane,
    Point,
    TwoDSketchSystem,
    Extrude,
    FeatureContainer,
    FeatureSystem,
)
from pancad.geometry.coordinate_system import Pose
from tests.testing_utils.sketch_gen import square, circle

def _square_extrude(length: float) -> Extrude:
    return Extrude.from_length(square(), length)

def _pose(x: float=0, y: float=0, z: float=0) -> Pose:
    return Pose(CoordinateSystem((x, y, z)))

EQ = True # Equal
NOT_EQ = False # Not Equal
CW = True # Clockwise
CCW = False # Counter-Clockwise

@pytest.mark.parametrize(
    "element, other, expected",
    [
        (Axis((0, 0, 0), (1, 0, 0)), Axis((0, 0, 0), (1, 0, 0)), EQ),
        (Axis((0, 0, 0), (1, 0, 0)), Axis((0, 1, 0), (1, 0, 0)), NOT_EQ),
        (Axis((0, 0, 0), (1, 0, 0)), Axis((0, 0, 0), (0, 1, 0)), NOT_EQ),
        (Circle((0, 0), 1), Circle((0, 0), 1), EQ),
        (Circle((0, 0), 1), Circle((0, 0), 2), NOT_EQ),
        (Circle((0, 0), 1), Circle((1, 1), 1), NOT_EQ),
        (
            CircularArc((0, 0), 1, (1, 0), (0, 1), CW),
            CircularArc((0, 0), 1, (1, 0), (0, 1), CW),
            EQ
        ),
        (
            CircularArc((0, 0), 1, (1, 0), (0, 1), CW),
            CircularArc((0, 0), 1, (1, 0), (0, 1), CCW),
            NOT_EQ
        ),
        (CoordinateSystem((0, 0)), CoordinateSystem((0, 0)), EQ),
        (CoordinateSystem((0, 0)), CoordinateSystem((1, 1)), NOT_EQ),
        (CoordinateSystem((0, 0, 0)), CoordinateSystem((0, 0, 0)), EQ),
        (CoordinateSystem((0, 0, 0)), CoordinateSystem((1, 1, 1)), NOT_EQ),
        (Line(Point(0, 0), (1, 0)), Line(Point(0, 0), (1, 0)), EQ),
        (Line(Point(0, 0), (1, 0)), Line(Point(0, 0), (1, 1)), NOT_EQ),
        (Line(Point(0, 0, 0), (1, 0, 0)), Line(Point(0, 0, 0), (1, 0, 0)), EQ),
        (Line(Point(0, 0, 0), (1, 0, 0)), Line(Point(0, 0, 0), (0, 1, 0)), NOT_EQ),
        (LineSegment((0, 0), (1, 1)), LineSegment((0, 0), (1, 1)), EQ),
        (LineSegment((0, 0, 0), (1, 1, 1)), LineSegment((0, 0, 0), (1, 1, 1)), EQ),
        (Plane((0, 0, 0), (1, 0, 0)), Plane((0, 0, 0), (1, 0, 0)), EQ),
        (Plane((0, 0, 0), (1, 0, 0)), Plane((0, 0, 0), (0, 1, 0)), NOT_EQ),
        (Point(0, 0, 0), Point(0, 0, 0), EQ),
        (Point(0, 0, 0), Point(1, 1, 1), NOT_EQ),
        (Ellipse.from_angle((0, 0), 2, 1, 0), Ellipse.from_angle((0, 0), 2, 1, 0), EQ),
        (Ellipse.from_angle((0, 0), 2, 1, 0), Ellipse.from_angle((1, 1), 2, 1, 0), NOT_EQ),
        (TwoDSketchSystem([Point(0,0,0)]), TwoDSketchSystem([Point(0,0,0)]), EQ),
        (TwoDSketchSystem([Point(0,0,0)]), TwoDSketchSystem([Point(1,1,1)]), NOT_EQ),
        (TwoDSketchSystem([Point(0,0,0)]), TwoDSketchSystem(), NOT_EQ),
        (_square_extrude(1), _square_extrude(1), EQ),
        (_square_extrude(1), _square_extrude(2), NOT_EQ),
        (FeatureContainer(), FeatureContainer(), EQ),
        (_pose(0, 0, 0), _pose(0, 0, 0), EQ),
        (_pose(0, 0, 0), _pose(1, 1, 1), NOT_EQ),
        (
            FeatureContainer(_pose(0,0,0), FeatureSystem(features=[square()])),
            FeatureContainer(_pose(0,0,0), FeatureSystem(features=[square()])),
            EQ,
        ),
        (
            FeatureContainer(_pose(0,0,0), FeatureSystem(features=[square()])),
            FeatureContainer(_pose(0,0,0), FeatureSystem(features=[circle()])),
            NOT_EQ,
        ),
    ]
)
def test_is_equal(element, other, expected):
    """Test that all is_equals on all Geometry and Features successfully 
    compares geometric equality by using known pairings of geometry and features.
    """
    assert element.is_equal(other) == expected


def test_rejects_duplicate_name_when_geometry_is_renamed() -> None:
    """Renaming geometry cannot make its name collide inside the system scope."""
    first = Point(0, 0, name="first")
    second = Point(1, 1, name="second")
    TwoDSketchSystem([first, second])

    with pytest.raises(ValueError, match="first"):
        second.name = "first"

    assert second.name == "second"


def test_rejects_duplicate_name_in_feature_geometry() -> None:
    """Feature-owned geometry keeps unique names even without an enclosing system."""
    feature = FeatureContainer()
    first = Point(0, 0, name="first")
    second = Point(1, 1, name="second")
    feature.feature_geometry.extend([first, second])

    with pytest.raises(ValueError, match="first"):
        second.name = "first"

    assert second.name == "second"
