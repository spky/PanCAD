"""Module for testing specifically the way that SketchGeometrySystems are
initialized and handle errors.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pancad.constants import ConstraintReference
from pancad.geometry.point import Point
from pancad.geometry.line_segment import LineSegment
from pancad.geometry.system import TwoDSketchSystem
from pancad.geometry.sketch import Sketch
from pancad.exceptions import (DupeUidError,
                               HasDependentsError,
                               MissingCADDependencyError)
from pancad.geometry.unique_lists import SketchGeometryList, SketchConstraintList
from pancad.constraints.snapto import Horizontal
from pancad.constraints.state_constraint import Coincident

from tests.testing_utils.thing_factory import make_feature

if TYPE_CHECKING:
    from pancad.abstract import AbstractConstraint, AbstractGeometry, PancadThing

    from tests._typing import FeatureSpec

    SequencePair = tuple[list[AbstractGeometry], list[AbstractConstraint]]


@pytest.fixture(name="sketch")
def fixture_sketch(feat_geo_sys_sample_samplesketches: FeatureSpec) -> Sketch:
    """A sketch generated from the samplesketch data file."""
    feature = make_feature(feat_geo_sys_sample_samplesketches)
    assert isinstance(feature, Sketch)
    return feature

@pytest.fixture(name="things")
def fixture_things(sketch: Sketch) -> list[PancadThing]:
    """The list of feature geometry, sketch parent geometry, and constraints in a sketch."""
    return [*sketch.feature_geometry,
            *sketch.geometry_system.geometry, *sketch.geometry_system.constraints]

@pytest.fixture(name="all_geometry")
def fixture_all_geometry(sketch: Sketch) -> list[AbstractGeometry]:
    """The list of all geometry inside a sketch."""
    geometry = [*sketch.feature_geometry, *sketch.geometry_system.geometry]
    return [child for geo in geometry for child in geo.children.values()]

class TestSketchElementFinding:
    """Tests for confirming that Sketch and its system can find all its elements using the sample
    sketch inside feat_geo_sys_sample.toml.
    """

    def test_feature_level_find(self, sketch: Sketch, things: list[PancadThing]) -> None:
        """Test that the sketch find method can find all the feature geometry, sketch constraints,
        and parent geometry.
        """
        for thing in things:
            assert thing.name is not None
            assert sketch.find(thing.name) == thing

    def test_system_level_find(self, sketch: Sketch) -> None:
        """Test that the geometry system find method can find all the sketch constraints and
        parent geometry.
        """
        for geometry in sketch.geometry_system.geometry:
            assert geometry.name is not None
            assert geometry == sketch.geometry_system.find(geometry.name)
        for constraint in sketch.geometry_system.constraints:
            assert constraint.name is not None
            assert constraint == sketch.geometry_system.find(constraint.name)

    def test_geometry_reference_find(self, sketch: Sketch,
                                     all_geometry: list[AbstractGeometry]) -> None:
        """Test that core and child geometry can be found using their references."""
        for geometry in all_geometry:
            prefix_name = geometry.parent.name if geometry.parent else geometry.name
            assert sketch.find(f"{prefix_name}::{geometry.self_reference}")

    def test_coordinate_system_find(self, sketch: Sketch) -> None:
        """Test that the sketch geometry's coordinate system is returned when coordinate system
        ConstraintReference is search for. For reference: This test is separated out since it's an
        unusual case of a feature having a 'weak' ConstraintReference. Sketches inherently have
        two coordinate systems (the one placing the sketch and the internal one), which is why
        feature locations use Poses rather than having their own coordinate system.
        """
        assert sketch.find(ConstraintReference.CS) == sketch.geometry_system.coordinate_system

# Setting up Fixtures
@pytest.fixture(name="empty_system")
def fixture_empty_system() -> TwoDSketchSystem:
    return TwoDSketchSystem()

@pytest.fixture(name="single_point")
def fixture_single_point() -> Point:
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
def fixture_system_with_constraints(system_just_geometry: TwoDSketchSystem,
                                    geometry_and_constraint_sequences: SequencePair
                                    ) -> TwoDSketchSystem:
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
    for i, geometry in enumerate(multiple_geometry_list):
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
