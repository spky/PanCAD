"""Module for testing specifically the way that SketchGeometrySystems are
initialized and handle errors.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pancad.constants import ConstraintReference, SketchConstraint as SC, QUAL_DELIM
from pancad.constraints._generator import make_constraint
from pancad.exceptions import DupeUidError, HasDependentsError, MissingCADDependencyError
from pancad.geometry.point import Point
from pancad.geometry.sketch import Sketch

from tests.testing_utils.thing_factory import make_feature

if TYPE_CHECKING:
    from pancad.abstract import AbstractGeometry, PancadThing

    from tests._typing import FeatureSpec


@pytest.fixture(name="sketch")
def fixture_sketch(feat_geo_sys_sample_samplesketches: FeatureSpec) -> Sketch:
    """A sketch generated from the sample sketch data file."""
    feature = make_feature(feat_geo_sys_sample_samplesketches)
    assert isinstance(feature, Sketch)
    return feature

@pytest.fixture(name="things")
def fixture_things(sketch: Sketch) -> list[PancadThing]:
    """The list of feature geometry, sketch parent geometry, and constraints inside the scope of a
    sample sketch."""
    return [*sketch.feature_geometry,
            *sketch.geometry_system.geometry, *sketch.geometry_system.constraints]

@pytest.fixture(name="all_geometry")
def fixture_all_geometry(sketch: Sketch) -> list[AbstractGeometry]:
    """The list of all geometry inside the scope of a sample sketch."""
    geometry = [*sketch.feature_geometry, *sketch.geometry_system.geometry]
    return [child for geo in geometry for child in geo.children.values()]

@pytest.fixture(name="empty_sketch")
def fixture_empty_sketch() -> Sketch:
    """An empty sketch."""
    return Sketch(name="initially_empty_sketch")

class TestSketchElementResolution:
    """Tests for confirming that Sketch and its system can resolve the names of all its elements
    using the sample sketches inside feat_geo_sys_sample.toml.
    """

    def test_feature_level_resolve(self, sketch: Sketch, things: list[PancadThing]) -> None:
        """Test that the sketch find method can find all the feature geometry, sketch constraints,
        and parent geometry.
        """
        for thing in things:
            assert thing.name is not None
            assert sketch.resolve(thing.name) == thing

    def test_system_level_resolve(self, sketch: Sketch) -> None:
        """Test that the geometry system find method can find all the sketch constraints and
        parent geometry.
        """
        for geometry in sketch.geometry_system.geometry:
            assert geometry.name is not None
            assert geometry == sketch.geometry_system.resolve(geometry.name)
        for constraint in sketch.geometry_system.constraints:
            assert constraint.name is not None
            assert constraint == sketch.geometry_system.resolve(constraint.name)

    def test_geometry_reference_resolve(self, sketch: Sketch,
                                        all_geometry: list[AbstractGeometry]) -> None:
        """Test that core and child geometry can be found using their references."""
        for geometry in all_geometry:
            prefix_name = geometry.parent.name if geometry.parent else geometry.name
            assert sketch.resolve(f"{prefix_name}{QUAL_DELIM}{geometry.self_reference}")

    def test_coordinate_system_find(self, sketch: Sketch) -> None:
        """Test that the sketch geometry's coordinate system is returned when coordinate system
        ConstraintReference is search for.

        .. note:: This test is separated out since it's an unusual case of a feature having a
            'weak' ConstraintReference. Sketches inherently have two coordinate systems (the one
            placing the sketch and the internal one), which is why feature locations use Poses
            rather than having their own coordinate system.
        """
        assert sketch.resolve(ConstraintReference.CS) == sketch.geometry_system.coordinate_system

    def test_parent_qual_name_resolution(self, sketch: Sketch, things: list[PancadThing]) -> None:
        """Test that the all sketch constraints and parent geometry qualified names can be
        resolved by the sketch.
        """
        for thing in things:
            # Split the first element off the name since it would be the sketch's name.
            assert sketch.resolve(thing.qualified_name.split(QUAL_DELIM, 1)[-1]) == thing

    def test_contains_qual_name(self, sketch: Sketch, things: list[PancadThing]) -> None:
        """Test that all sketch constraints and parent geometry qualified names return True when
        checked for containment.
        """
        for thing in things:
            # Split off first qualified name prefix since the sketch doesn't contain itself.
            assert thing.qualified_name.split(QUAL_DELIM, 1)[-1] in sketch

    def test_rejects_duplicate_name_when_geometry_is_added(self,
                                                            empty_sketch: Sketch) -> None:
        """A geometry name must be unique before the element enters the sketch scope."""
        first = Point(0, 0, name="shared_name")
        duplicate = Point(1, 1, name="shared_name")
        empty_sketch.geometry_system.geometry.append(first)

        with pytest.raises(ValueError, match="shared_name"):
            empty_sketch.geometry_system.geometry.append(duplicate)

        assert list(empty_sketch.geometry_system.geometry) == [first]
        assert duplicate.system is None

    def test_rejects_name_shared_by_geometry_and_constraint(self,
                                                             empty_sketch: Sketch) -> None:
        """Geometry and constraints share one namespace in a sketch system."""
        point = Point(0, 0, name="shared_name")
        empty_sketch.geometry_system.geometry.append(point)
        constraint = make_constraint(SC.FIXED, point, name="shared_name")

        with pytest.raises(ValueError, match="shared_name"):
            empty_sketch.geometry_system.constraints.append(constraint)

        assert not empty_sketch.geometry_system.constraints
        assert constraint.system is None


class TestSketchGeometryList:
    """Tests for confirming that Sketch system geometry lists are correctly constructed,
    modifications are possible and invalid sketch states are checked for.
    """

    def test_system_assignment(self, sketch: Sketch) -> None:
        """Test that all sample sketch geometry had their system and feature assigned when they
        were added to the sketch.
        """
        for geometry in sketch.geometry_system.geometry:
            assert geometry.system == sketch.geometry_system
            assert geometry.feature == sketch

    def test_index(self, sketch: Sketch) -> None:
        """Test that all sample sketch geometry list specific indicies can be returned."""
        for i, geometry in enumerate(sketch.geometry_system.geometry):
            assert sketch.geometry_system.geometry.index(geometry) == i

    def test_append(self, empty_sketch: Sketch) -> None:
        """Test that a geometry element can be appended to a sketch's geometry list."""
        point = Point(1, 1)
        empty_sketch.geometry_system.geometry.append(point)
        assert empty_sketch.geometry_system.geometry[0] == point

    def test_duplicated_geometry(self, empty_sketch: Sketch) -> None:
        """Test that adding the same geometry twice to a sketch raises a DupeUidError."""
        point = Point(1, 1)
        empty_sketch.geometry_system.geometry.append(point)
        with pytest.raises(DupeUidError):
            empty_sketch.geometry_system.geometry.append(point)

    def test_del(self, empty_sketch: Sketch) -> None:
        """Test that deleting geometry from the custom unique list is possible and that the
        removed geometry's system/feature properties are set to None.
        """
        point = Point(1, 1)
        empty_sketch.geometry_system.geometry.append(point)
        del empty_sketch.geometry_system.geometry[0]
        assert len(empty_sketch.geometry_system.geometry) == 0
        assert point.system is None
        assert point.feature is None

    def test_del_with_constraints(self, empty_sketch: Sketch) -> None:
        """Test that deleting geometry that still has constraints raises a HasDependentsError."""
        system, point = empty_sketch.geometry_system, Point(0, 0)
        system.geometry.append(point)
        system.constrain(SC.COINCIDENT, empty_sketch.geometry_system.origin, point)
        with pytest.raises(HasDependentsError):
            del system.geometry[0]

class TestSketchConstraintList:
    """Tests confirming that Sketch system constraints and correctly constructed, modifications
    are possible, and invalid sketch states are checked for.
    """

    def test_add_without_dependencies(self, empty_sketch: Sketch) -> None:
        """Test that adding a constraint to a system without the constrained geometry already in
        the system raises a missingcaddependencyerror.
        """
        constraint = make_constraint(SC.COINCIDENT,
                                     Point(0, 0), empty_sketch.geometry_system.origin)
        with pytest.raises(MissingCADDependencyError):
            empty_sketch.geometry_system.constraints.append(constraint)

    def test_duplicated_constraint(self, empty_sketch: Sketch) -> None:
        """Test that adding the same constraint twice to a sketch raises a DupeUidError."""
        system, point = empty_sketch.geometry_system, Point(0, 0)
        system.geometry.append(point)
        system.constrain(SC.COINCIDENT, empty_sketch.geometry_system.origin, point)
        with pytest.raises(DupeUidError):
            system.constraints.append(system.constraints[0])
