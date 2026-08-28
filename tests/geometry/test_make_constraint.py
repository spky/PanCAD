"""Tests for the generalized make_constraint function used as a one stop shop to create pancad
constraints.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pancad.constants import SketchConstraint as SC
from pancad.constraints.distance import Angle, AbstractDistance
from pancad.constraints.state_constraint import AbstractStateConstraint
from pancad.constraints._generator import make_constraint
from pancad.geometry.point import Point
from pancad.geometry.line import Line

if TYPE_CHECKING:
    from pytest import FixtureRequest

    from pancad.constraints.snapto import AbstractSnapTo, AbstractSingleSnapTo

    NonValueConstraint = AbstractStateConstraint | AbstractSnapTo | AbstractSingleSnapTo

@pytest.fixture(name="point_duo", params=[[(0, 0), (1, 1)], [(0, 0, 0), (1, 1, 1)]])
def fixture_point_duo(request: FixtureRequest) -> tuple[Point, Point]:
    """Returns a pair of same dimension pancad Points."""
    points: tuple[Point, ...] = tuple()
    for vector in request.param:
        points = points + (Point(vector),)
    assert len(points) == 2
    return points

@pytest.fixture(name="line_2d_duo", params=[[(1, 0), (1, 1)]])
def fixture_line_2d_duo(request: FixtureRequest) -> tuple[Line, Line]:
    """Returns a pair of 2 dimensional pancad Lines."""
    lines: tuple[Line, ...] = tuple()
    for direction in request.param:
        lines = lines + (Line(Point([0] * len(direction)), direction),)
    assert len(lines) == 2
    return lines

class TestNominal:
    """Tests for make_constraint's inputs as they are expected to be input. These tests are also
    used to check that static type checking the make_constraint function works in all cases.
    """

    @pytest.mark.parametrize("type_", [SC.COINCIDENT, "coincident"])
    def test_coincident(self, type_: SC | str, point_duo: tuple[Point, Point]) -> None:
        """Tests that it's possible to generate a Coincident constraint between Points."""
        constraints: list[NonValueConstraint] = [
            make_constraint(type_, *point_duo),
            make_constraint(type_, *point_duo, uid=None),
            make_constraint(type_, *point_duo, system=None),
            make_constraint(type_, *point_duo, uid=None, system=None),
        ]
        for constraint in constraints:
            assert isinstance(constraint, AbstractStateConstraint)

    @pytest.mark.parametrize("type_", [SC.ANGLE, "angle"])
    def test_angle(self, type_: SC | str, line_2d_duo: tuple[Line, Line]) -> None:
        """Tests that it's possible to generate a Angle constraint between 2D Lines."""
        constraints: list[Angle] = [
            make_constraint(type_, *line_2d_duo, value=45, quadrant=1),
            make_constraint(type_, *line_2d_duo, value=45, quadrant=1, is_radians=False),
            make_constraint(type_, *line_2d_duo, value=45, quadrant=1, uid=None),
            make_constraint(type_, *line_2d_duo, value=45, quadrant=1, uid=None,
                            is_radians=False),
            make_constraint(type_, *line_2d_duo, value=45, quadrant=1, system=None),
            make_constraint(type_, *line_2d_duo, value=45, quadrant=1, system=None,
                            is_radians=False),
            make_constraint(type_, *line_2d_duo, value=45, quadrant=1, uid=None, system=None),
            make_constraint(type_, *line_2d_duo, value=45, quadrant=1, uid=None, system=None,
                            is_radians=False),
        ]
        for constraint in constraints:
            assert isinstance(constraint, Angle)

    @pytest.mark.parametrize("type_", [SC.DISTANCE, "distance"])
    def test_distance(self, type_: SC | str, point_duo: tuple[Point, Point]) -> None:
        """Tests that it's possible to generate a Distance constraint between Points."""
        constraints: list[AbstractDistance] = [
            make_constraint(type_, *point_duo, value=0),
            make_constraint(type_, *point_duo, value=0, unit=None),
            make_constraint(type_, *point_duo, value=0, unit="in"),
            make_constraint(type_, *point_duo, value=0, uid=None),
            make_constraint(type_, *point_duo, value=0, unit="in", uid=None),
        ]
        for constraint in constraints:
            assert isinstance(constraint, AbstractDistance)

class TestValueErrors:
    """Test cases where make_constraint will raise ValueErrors due to incorrect input combos."""

    def test_distance(self, point_duo: tuple[Point, Point]) -> None:
        """Test that not providing a value for a distance constraint raises an error."""
        with pytest.raises(ValueError, match="'value' was not provided"):
            make_constraint("distance", *point_duo)

    @pytest.mark.parametrize(
        "value, quadrant, err_names",
        [(45, None, "quadrant"), (None, 45, "value"), (None, None, "value, quadrant")]
    )
    def test_angle(self, value: float | None, quadrant: int | None, err_names: str,
                   line_2d_duo: tuple[Line, Line]) -> None:
        """Test that not providing a value or a quadrant and angle constraint raises an error."""
        with pytest.raises(ValueError, match=f"'{err_names}' was not provided"):
            make_constraint("angle", *line_2d_duo, value=value, quadrant=quadrant)
