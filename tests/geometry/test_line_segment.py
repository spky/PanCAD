"""Tests for pancad's LineSegment class.

.. note:: Initialization type testing is not performed on LineSegment's init since both arguments
    are immediately turned into Points so adding those tests would be redundant.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pancad.geometry.line_segment import LineSegment
from pancad.utils import trigonometry as trig, solvers

from tests.testing_utils.data_collectors import make_geometry_sample_input

if TYPE_CHECKING:
    from typing import Literal, Type

    from pancad.utils.pancad_types import SpaceVector, Space2DVector
    from tests._typing import GeometrySampleData

def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Generates tests specific to LineSegment testing in this file from sample data."""
    if "pla_data" in metafunc.fixturenames:
        ids, sample_data = make_geometry_sample_input("data_line_segment_angle_segments")
        metafunc.parametrize("pla_data", sample_data, ids=ids)

@pytest.fixture(name="line_segment")
def fixture_line_segment(data_line_segment_start_end_segments: GeometrySampleData,
                         request: pytest.FixtureRequest) -> LineSegment:
    """Returns a sample LineSegment read from a test data file to test with."""
    id_ = request.node.callspec.id
    vectors = data_line_segment_start_end_segments["vectors"]
    if "start_end_segments" in id_:
        return LineSegment(vectors["start"], vectors["end"])
    raise ValueError(f"Unexpected data group: {id_}")

@pytest.fixture(name="pla_start")
def fixture_pla_start(pla_data: GeometrySampleData) -> SpaceVector:
    """The start point of a sample line created from a point, length, and angle(s)."""
    return pla_data["vectors"]["start"]

@pytest.fixture(name="pla_vector")
def fixture_pla_vector(pla_data: GeometrySampleData) -> SpaceVector:
    """The polar/spherical vector from the start to the end point of a sample line created from a
    point, length, and angle(s).
    """
    scalars = pla_data["scalars"]
    if "theta" in scalars:
        return (scalars["length"], scalars["phi"], scalars["theta"])
    return (scalars["length"], scalars["phi"])

@pytest.fixture(name="pla_expected")
def fixture_pla_expected(pla_data: GeometrySampleData) -> LineSegment:
    """The expected LineSegment created from the start and vector data of a sample line created
    from a point, length, and angle(s).
    """
    return LineSegment(pla_data["vectors"]["start"], pla_data["vectors"]["end"])

class TestProperties:
    """Tests for reading and modifying LineSegment properties."""

    def test_direction(self, line_segment: LineSegment) -> None:
        """Test that the LineSegment's direction is the unit vector pointing from the start point
        to the end point
        """
        expected = trig.get_unit_vector(line_segment.end - line_segment.start)
        np.testing.assert_allclose(line_segment.direction, expected)

    def test_update(self) -> None:
        """Test that LineSegment can be updated to match another LineSegment."""
        segment = LineSegment((0, 0, 0), (1, 0, 0))
        new = LineSegment((1, 1, 1), (2, 2, 2))
        segment.update(new)
        assert segment.is_equal(new)

class TestFromPointLengthAngle:
    """Tests for initializing a LineSegment from a combination of a point, length, and angles."""

    def test_tuple_input(self, pla_start: SpaceVector, pla_vector: SpaceVector,
                         pla_expected: LineSegment) -> None:
        """Test creating a LineSegment with a tuple input for the polar/spherical vector."""
        segment = LineSegment.from_point_length_angle(pla_start, pla_vector)
        assert segment.is_equal(pla_expected)

    def test_float_input(self, pla_start: SpaceVector, pla_vector: SpaceVector,
                         pla_expected: LineSegment) -> None:
        """Test creating a LineSegment with multiple floats for the polar/spherical vector."""
        segment = LineSegment.from_point_length_angle(pla_start, *pla_vector)
        assert segment.is_equal(pla_expected)

    @pytest.mark.parametrize("start, vector",
                             [pytest.param((0, 0, 0), (1, 2)), pytest.param((0, 0), (1, 2, 3))])
    def test_dimension_mismatch(self, start: SpaceVector, vector: SpaceVector) -> None:
        """Test that providing start and vector components of differing dimensions raises an
        error.
        """
        with pytest.raises(ValueError, match="^start/components"):
            LineSegment.from_point_length_angle(start, vector)

class TestSolvers:
    """Tests for geometry property solving methods separate from LineSegment."""

    def test_get_length(self, line_segment: LineSegment) -> None:
        """Test that the length of the line in specified directions can be solved for."""
        letters: tuple[Literal["x", "y", "z"], ...] = ("x", "y", "z")
        for start_c, end_c, letter in zip(line_segment.start, line_segment.end, letters):
            assert solvers.get_length(line_segment, letter) == abs(start_c - end_c) # Specified
        direction_vector = line_segment.start - line_segment.end
        assert solvers.get_length(line_segment) == math.hypot(*direction_vector) # Overall

    @pytest.mark.parametrize(
        "segment, along, msg",
        [
            pytest.param(LineSegment((0, 0), (1, 1)), "z",
                         r"Expected one of \['x', 'y'\]", id="2dZ"),
            pytest.param(LineSegment((0, 0, 0), (1, 1, 1)),
                         "W", r"Expected one of \['x', 'y', 'z'\]", id="3dW"),
        ]
    )
    def test_get_length_excs(self, segment: LineSegment, along: str, msg: str) -> None:
        """Test that the exceptions of solver.get_length activate and provide the
        right message.
        """
        with pytest.raises(TypeError, match=msg):
            solvers.get_length(segment, along) # type: ignore # Testing if user ignores types.

    @pytest.mark.parametrize(
        "segment, value, from_, along, expected",
        [
            pytest.param(LineSegment((0, 0), (1, 1)), np.sqrt(8), "start", None,
                         LineSegment((0, 0), (2, 2)), id="2dStartNone2"),
            pytest.param(LineSegment((0, 0), (1, 1)), np.sqrt(8), "end", None,
                         LineSegment((-1, -1), (1, 1)), id="2dEndNone2"),
            pytest.param(LineSegment((0, 0), (1, 0)), 2, "start", "y",
                         LineSegment((0, 0), (1, 2)), id="2dStartY2"),
            pytest.param(LineSegment((0, 0), (1, 0)), 2, "end", "y",
                         LineSegment((0, -2), (1, 0)), id="2dEndY2"),
            pytest.param(LineSegment((0, 0, 0), (1, 1, 1)), np.sqrt(12), "start", None,
                         LineSegment((0, 0, 0), (2, 2, 2)), id="3dStartNone2"),
            pytest.param(LineSegment((0, 0, 0), (1, 1, 1)), np.sqrt(12), "end", None,
                         LineSegment((-1, -1, -1), (1, 1, 1)), id="3dEndNone2"),
        ]
    )
    def test_set_length(self,
                        segment: LineSegment,
                        value: float,
                        from_: Literal["start", "end"],
                        along: Literal["x", "y", "z"] | None,
                        expected: LineSegment) -> None:
        """Test that solvers.set_length can set the length of a line segment from
        start or end, along the line, and along specified axes.
        """
        solvers.set_length(segment, value, from_, along)
        assert segment.is_equal(expected)

    @pytest.mark.parametrize(
        "segment, value, from_, along, error_type, msg",
        [
            pytest.param(LineSegment((0, 0), (1, 1)), 0, "start", None,
                         ValueError, "Length cannot be set to 0", id="ZeroLength"),
            pytest.param(LineSegment((0, 0), (1, 1)), 1, "Fake", None,
                         TypeError, "^Unexpected from_", id="FromFake"),
            pytest.param(LineSegment((0, 0), (1, 1)), 1, "start", "W",
                         TypeError, "^Unexpected along", id="AlongW"),
        ]
    )
    def test_set_length_excs(self,
                             segment: LineSegment,
                             value: float,
                             from_: str,
                             along: str,
                             error_type: Type[Exception],
                             msg: str) -> None:
        """Test that the exceptions of solver.set_length activate and provide the right message.
        """
        with pytest.raises(error_type, match=msg):
            solvers.set_length(segment, value, from_, along) # type: ignore # User Type Miss Test

    @pytest.mark.parametrize(
        "segment, bottom_left, top_right",
        [
            (LineSegment((0, 0), (1, -1)), (0, -1), (1, 0)),
            (LineSegment((0, 0), (1, 1)), (0, 0), (1, 1)),
            (LineSegment((1, 1), (0, 0)), (0, 0), (1, 1)),
            (LineSegment((-1, -1), (0, 0)), (-1, -1), (0, 0)),
        ]
    )
    def test_get_fit_box(self, segment: LineSegment,
                         bottom_left: Space2DVector, top_right: Space2DVector) -> None:
        """Test that solvers can return the expected 2d box points for a LineSegment."""
        expected = (bottom_left, top_right)
        assert np.array(solvers.get_fit_box(segment)) == pytest.approx(np.array(expected))

    @pytest.mark.parametrize(
        "segment, error_type, msg",
        [
            (LineSegment((0, 0, 0), (1, 1, 1)),
             NotImplementedError, "^Fit boxes of 3D geometry are not"),
        ]
    )
    def test_get_fit_box_excs(self, segment: LineSegment,
                              error_type: Type[Exception], msg: str) -> None:
        """Test that the exceptions of solver.get_fit_box activate and provide the
        right message for a LineSegment.
        """
        with pytest.raises(error_type, match=msg):
            solvers.get_fit_box(segment)
