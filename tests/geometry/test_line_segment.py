"""Tests for pancad's LineSegment class"""
from __future__ import annotations

import itertools
import math
import unittest
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pancad.geometry.point import Point
from pancad.geometry.line import Line
from pancad.geometry.line_segment import LineSegment
from pancad.utils import trigonometry as trig, solvers

if TYPE_CHECKING:
    from typing import Literal

    from tests._typing import GeometrySampleData

@pytest.fixture(name="line_segment")
def fixture_line_segment(data_line_segment: GeometrySampleData,
                         request: pytest.FixtureRequest) -> LineSegment:
    """Returns a sample LineSegment read from a test data file to test with."""
    id_ = request.node.callspec.id
    vectors = data_line_segment["vectors"]
    if "start_end_segments" in id_:
        return LineSegment(vectors["start"], vectors["end"])
    raise ValueError(f"Unexpected data group: {id_}")

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
def test_set_length(segment, value, from_, along, expected):
    """Test that solvers.set_length can set the length of a line segment from
    start or end, along the line, and along specified axes.
    """
    solvers.set_length(segment, value, from_, along)
    assert segment.is_equal(expected)

@pytest.mark.parametrize(
    "segment, value, along, from_, error_type, msg",
    [
        pytest.param(LineSegment((0, 0), (1, 1)), 0, "start", None,
                     ValueError, "Length cannot be set to 0", id="ZeroLength"),
        pytest.param(LineSegment((0, 0), (1, 1)), 1, "Fake", None,
                     TypeError, "^Unexpected from_", id="FromFake"),
        pytest.param(LineSegment((0, 0), (1, 1)), 1, "start", "W",
                     TypeError, "^Unexpected along", id="AlongW"),
    ]
)
def test_set_length_excs(segment, value, along, from_, error_type, msg):
    """Test that the exceptions of solver.set_length activate and provide the
    right message.
    """
    with pytest.raises(error_type, match=msg):
        solvers.set_length(segment, value, along, from_)

@pytest.mark.parametrize(
    "segment, expected",
    [
        (LineSegment((0, 0), (1, -1)), ((0, -1), (1, 0))),
        (LineSegment((0, 0), (1, 1)), ((0, 0), (1, 1))),
        (LineSegment((1, 1), (0, 0)), ((0, 0), (1, 1))),
        (LineSegment((-1, -1), (0, 0)), ((-1, -1), (0, 0))),
    ]
)
def test_get_fit_box(segment, expected):
    assert np.array(solvers.get_fit_box(segment)) == pytest.approx(np.array(expected))

@pytest.mark.parametrize(
    "segment, error_type, msg",
    [
        (LineSegment((0, 0, 0), (1, 1, 1)),
         NotImplementedError, "^Fit boxes of 3D geometry are not"),
    ]
)
def test_get_fit_box_excs(segment, error_type, msg):
    """Test that the exceptions of solver.get_fit_box activate and provide the
    right message for a LineSegment.
    """
    with pytest.raises(error_type, match=msg):
        solvers.get_fit_box(segment)

class TestLineSegmentInit2d(unittest.TestCase):
    def setUp(self):
        self.pt_a, self.pt_b = (0, 0), (1, 1)
        self.check_features = (Point(self.pt_a), Point(self.pt_b),
                               Line.from_two_points(self.pt_a, self.pt_b))

    def test_init_two_points(self):
        line_seg = LineSegment(Point(self.pt_a), Point(self.pt_b), uid="test")
        test_features = (line_seg.start, line_seg.end,
                         line_seg.get_line())
        for check, test in zip(self.check_features, test_features):
            with self.subTest(test=test, check=check):
                self.assertTrue(test.is_equal(check))

    def test_init_two_tuples(self):
        line_seg = LineSegment(self.pt_a, self.pt_b)
        test_features = (line_seg.start, line_seg.end,
                         line_seg.get_line())
        for check, test in zip(self.check_features, test_features):
            with self.subTest(test=test, check=check):
                self.assertTrue(test.is_equal(check))

class TestLineSegmentInit3d(unittest.TestCase):
    def setUp(self):
        self.pt_a, self.pt_b = (0, 0, 0), (1, 1, 1)
        self.check_features = (Point(self.pt_a), Point(self.pt_b),
                               Line.from_two_points(self.pt_a, self.pt_b))

    def test_init_two_points(self):
        line_seg = LineSegment(Point(self.pt_a), Point(self.pt_b))
        test_features = (line_seg.start, line_seg.end,
                         line_seg.get_line())
        for check, test in zip(self.check_features, test_features):
            with self.subTest(test=test, check=check):
                self.assertTrue(test.is_equal(check))

    def test_init_two_tuples(self):
        line_seg = LineSegment(self.pt_a, self.pt_b)
        test_features = (line_seg.start, line_seg.end,
                         line_seg.get_line())
        for check, test in zip(self.check_features, test_features):
            with self.subTest(test=test, check=check):
                self.assertTrue(test.is_equal(check))

class TestLineSegmentFromPointLengthAngle(unittest.TestCase):
    def test_init_polar_vector(self):
        point = (0, 0)
        polar = (2, math.radians(45))
        test_ls = LineSegment.from_point_length_angle(point, polar)
        expected_ls = LineSegment(point, (math.sqrt(2), math.sqrt(2)))
        self.assertTrue(test_ls.is_equal(expected_ls))

    def test_init_polar_float(self):
        point = (0, 0)
        polar = (2, math.radians(45))
        test_ls = LineSegment.from_point_length_angle(point, *polar)
        expected_ls = LineSegment(point, (math.sqrt(2), math.sqrt(2)))
        self.assertTrue(test_ls.is_equal(expected_ls))

    def test_init_spherical_vector(self):
        point = (0, 0, 0)
        spherical = (2, math.radians(45), math.radians(90))
        test_ls = LineSegment.from_point_length_angle(point, spherical)
        expected_ls = LineSegment(point, (math.sqrt(2), math.sqrt(2), 0))
        self.assertTrue(test_ls.is_equal(expected_ls))

    def test_init_spherical_float(self):
        point = (0, 0, 0)
        spherical = (2, math.radians(45), math.radians(90))
        test_ls = LineSegment.from_point_length_angle(point, *spherical)
        expected_ls = LineSegment(point, (math.sqrt(2), math.sqrt(2), 0))
        self.assertTrue(test_ls.is_equal(expected_ls))

class TestLineSegmentFromPointLengthAngleExceptions(unittest.TestCase):

    def setUp(self):
        self.pt2d = (0, 0)
        self.polar = (2, math.radians(45))
        self.pt3d = self.pt2d + (0,)
        self.spherical = self.polar + (math.radians(90),)
        self.length, self.phi, self.theta = self.spherical

    def test_polar_vector_phi(self):
        with self.assertRaises(TypeError):
            LineSegment.from_point_length_angle(self.pt2d, self.polar, 3)

    def test_polar_vector_phi_theta(self):
        with self.assertRaises(TypeError):
            LineSegment.from_point_length_angle(self.pt2d, self.polar, 3, 3)

    def test_spherical_vector_phi(self):
        with self.assertRaises(TypeError):
            LineSegment.from_point_length_angle(self.pt3d, self.spherical, 3)

    def test_spherical_vector_phi_theta(self):
        with self.assertRaises(TypeError):
            LineSegment.from_point_length_angle(self.pt3d, self.spherical, 3, 3)

    def test_length_no_phi(self):
        with self.assertRaises(TypeError):
            LineSegment.from_point_length_angle(self.pt2d, self.length)

    def test_dimension_mismatch_3to2(self):
        with self.assertRaises(ValueError):
            LineSegment.from_point_length_angle(self.pt3d, self.polar)

    def test_dimension_mismatch_2to3(self):
        with self.assertRaises(ValueError):
            LineSegment.from_point_length_angle(self.pt2d, self.spherical)

class TestLineSegmentGetters(unittest.TestCase):

    def setUp(self):
        lines = [
            ((0, 0), (1, 1)),
            ((0, 1), (1, 2)),
            ((0, 0, 0), (1, 1, 1)),
        ]
        test_lines = [LineSegment(*line) for line in lines]
        directions = [
            (1, 1),
            (1, 1),
            (1, 1, 1),
        ]
        directions = [
            trig.to_1d_tuple(trig.get_unit_vector(d)) for d in directions
        ]
        lengths = [
            math.hypot(1, 1),
            math.hypot(1, 1),
            math.hypot(1, 1, 1),
        ]
        axis_lengths = [
            (1, 1, None),
            (1, 1, None),
            (1, 1, 1),
        ]
        self.direction_tests = list(zip(test_lines, directions))
        self.length_tests = list(zip(test_lines, lengths))
        self.axis_length_tests = list(zip(test_lines, axis_lengths))

    def test_direction_getter(self):
        for segment, direction in self.direction_tests:
            with self.subTest(line_segment=segment, direction=direction):
                np.testing.assert_allclose(segment.direction, direction)

class TestLineSegmentUpdate(unittest.TestCase):

    def test_update(self):
        ls = LineSegment((0, 0, 0), (1, 0, 0))
        new = LineSegment((1, 1, 1), (2, 2, 2))
        ls.update(new)
        self.assertTrue(ls.is_equal(new))

if __name__ == "__main__":
    unittest.main()
