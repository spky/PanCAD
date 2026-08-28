"""Module for testing specifically the way that SketchGeometrySystems are 
initialized and handle errors.
"""
from __future__ import annotations

import pytest
from typing import TYPE_CHECKING

from pancad.geometry.point import Point
from pancad.geometry.line_segment import LineSegment
from pancad.geometry.system import TwoDSketchSystem
from pancad.exceptions import (DupeUidError,
                               HasDependentsError,
                               MissingCADDependencyError)
from pancad.geometry.unique_lists import SketchGeometryList, SketchConstraintList
from pancad.constraints.snapto import Horizontal
from pancad.constraints.state_constraint import Coincident

if TYPE_CHECKING:
    from pancad.abstract import AbstractConstraint, AbstractGeometry

    SequencePair = tuple[list[AbstractGeometry], list[AbstractConstraint]]

# Setting up Fixtures
@pytest.fixture(name="empty_system")
def fixture_empty_system() -> TwoDSketchSystem:
    return TwoDSketchSystem()

@pytest.fixture(name="single_point")
def single_point() -> Point:
    return Point(0, 0)

@pytest.fixture(name="empty_geometry_list")
def fixture_empty_geometry_list(empty_system: TwoDSketchSystem) -> SketchGeometryList:
    return empty_system.geometry

@pytest.fixture(name="empty_constraint_list")
def fixture_empty_constraint_list(empty_system: TwoDSketchSystem) -> SketchConstraintList:
    return empty_system.constraints

@pytest.fixture(name="list_of_points")
def fixture_list_of_points() -> list[Point]:
    return [Point(0, 0), Point(1, 1), Point(2, 2)]

@pytest.fixture(name="multiple_geometry_list", params=["list_of_points"])
def fixture_multiple_geometry_list(empty_geometry_list: SketchGeometryList,
                                   request: pytest.FixtureRequest) -> SketchGeometryList:
    empty_geometry_list.extend(request.getfixturevalue(request.param))
    return empty_geometry_list

@pytest.fixture(name="horizontal_line_segment")
def fixture_horizontal_line_segment() -> tuple[list[LineSegment], list[Horizontal]]:
    line = LineSegment((0, 0), (1, 0))
    return [line], [Horizontal(line)]

@pytest.fixture(name="line_segment_coincident_with_origin")
def fixture_line_segment_coincident_with_origin(empty_system: TwoDSketchSystem
                                                ) -> tuple[list[LineSegment], list[Coincident]]:
    line = LineSegment((0, 0), (1, 0))
    return [line], [Coincident(line, empty_system.origin)]

@pytest.fixture(
    name="geometry_and_constraint_sequences",
    params = [
        "horizontal_line_segment", "line_segment_coincident_with_origin",
    ]
)
def fixture_geometry_and_constraint_sequences(request: pytest.FixtureRequest) -> SequencePair:
    value = request.getfixturevalue(request.param)
    return value

@pytest.fixture(name="system_just_geometry")
def fixture_system_just_geometry(empty_system: TwoDSketchSystem,
                                 geometry_and_constraint_sequences: SequencePair
                                 ) -> TwoDSketchSystem:
    geometry, _ = geometry_and_constraint_sequences
    empty_system.geometry.extend(geometry)
    return empty_system

@pytest.fixture(name="system_with_constraints")
def system_with_constraints(system_just_geometry: TwoDSketchSystem,
                            geometry_and_constraint_sequences: SequencePair) -> TwoDSketchSystem:
    """Systems where all geometry in the list has at least one constraint on it.
    """
    _, constraints = geometry_and_constraint_sequences
    system_just_geometry.constraints.extend(constraints)
    return system_just_geometry

# Testing GeometryList
def test_system_coordinate_system_in_check(empty_system: TwoDSketchSystem) -> None:
    assert empty_system in empty_system.geometry

def test_append_geometry(empty_geometry_list: SketchGeometryList,
                         single_point: Point) -> None:
    empty_geometry_list.append(single_point)
    assert empty_geometry_list[0] is single_point

def test_duped_geometry_list(empty_geometry_list: SketchGeometryList,
                             single_point: Point) -> None:
    empty_geometry_list.append(single_point)
    with pytest.raises(DupeUidError):
        empty_geometry_list.append(single_point)

def test_delete_geometry_in_empty(empty_geometry_list: SketchGeometryList,
                                  single_point: Point) -> None:
    empty_geometry_list.append(single_point)
    del empty_geometry_list[0]
    assert len(empty_geometry_list) == 0

def test_geometry_list_index(multiple_geometry_list: SketchGeometryList) -> None:
    for i in range(len(multiple_geometry_list)):
        geometry = multiple_geometry_list[i]
        assert multiple_geometry_list.index(geometry) == i

def test_assign_system(system_just_geometry: TwoDSketchSystem) -> None:
    for geometry in system_just_geometry.geometry:
        assert geometry.system is system_just_geometry
        assert geometry.feature is None

def test_delete_geometry_system(system_just_geometry: TwoDSketchSystem) -> None:
    geometry = system_just_geometry.geometry[0]
    del system_just_geometry.geometry[0]
    assert geometry.system is None
    assert geometry.feature is None

def test_delete_geometry_with_constraints(system_with_constraints: TwoDSketchSystem) -> None:
    with pytest.raises(HasDependentsError):
        del system_with_constraints.geometry[0]


# Testing ConstraintList
def test_add_constraint_without_dependencies(empty_constraint_list: SketchConstraintList,
                                             geometry_and_constraint_sequences: SequencePair
                                             ) -> None:
    _, constraints = geometry_and_constraint_sequences
    with pytest.raises(MissingCADDependencyError):
        empty_constraint_list.append(constraints[0])

def test_add_duped_constraint(geometry_and_constraint_sequences: SequencePair,
                              system_with_constraints: TwoDSketchSystem) -> None:
    _, constraints = geometry_and_constraint_sequences
    with pytest.raises(DupeUidError):
        system_with_constraints.constraints.append(constraints[0])