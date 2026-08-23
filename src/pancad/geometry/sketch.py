"""A module providing a class to represent sketches in 3D space. pancad defines a
sketch as a set of 2D geometry on a coordinate system's plane oriented in 3D
space. pancad's sketch definition aims to be as general as possible, so the
base implementation of this class does not include appearance information since
that is application specific.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from pancad.abstract import AbstractFeature
from pancad.constants import SketchConstraint
from pancad.geometry.coordinate_system import Pose
from pancad.geometry.unique_lists import FeatureGeometryList
from pancad.geometry.system import TwoDSketchSystem, FeatureSystem
from pancad.utils.initialize import get_pancad_config

if TYPE_CHECKING:
    from uuid import UUID
    from numbers import Real
    from collections.abc import Sequence
    from typing import Self

    from pancad.abstract import (
        AbstractGeometry, AbstractConstraint, PancadThing, AbstractFeatureSystem
    )
    from pancad.geometry.line import Line
    from pancad.geometry.plane import Plane
    from pancad.geometry.point import Point

DEFAULT_NAME = get_pancad_config()["features"]["default_names"]["sketch"]



class Sketch(AbstractFeature):
    """A class representing a sketch feature that places a 2D system of
    geometry/constraints onto a plane in 3D space.

    :param geometry_system: The TwoDSketchSystem containing the geometry and
        constraints inside the Sketch.
    :param pose: The location and orientation of the Sketch.
    :param system: The feature system that the Sketch's pose is defined in.
    :param name: The name of the Sketch that will be used wherever a CAD
        application requires a human-readable name for the sketch element.
    :param uid: The unique id of the Sketch. Defaults to None.
    """
    def __init__(self, geometry_system: TwoDSketchSystem | None=None, pose: Pose | None=None, *,
                 system: AbstractFeatureSystem | None=None,
                 name: str=DEFAULT_NAME,
                 uid: str | UUID | None=None):
        super().__init__(system, name)
        self.uid = uid
        if pose is None:
            pose = Pose.from_yaw_pitch_roll((0, 0, 0), 0, 0, 0)
        if geometry_system is None:
            geometry_system = TwoDSketchSystem()
        self._pose = pose
        self._geometry_system = geometry_system
        self._feature_geometry = FeatureGeometryList(
            self, [self._pose, self._geometry_system]
        )

    # Properties #
    @property
    def pose(self) -> Pose:
        """The location and orientation of the 2D sketch coordinate system."""
        return self._pose
    @pose.setter
    def pose(self, value: Pose) -> None:
        self._pose.update(value)

    @property
    def feature_geometry(self) -> FeatureGeometryList:
        """The geometry directly owned by this Sketch. Usually its Pose and GeometrySystem."""
        return self._feature_geometry

    @property
    def geometry_system(self) -> TwoDSketchSystem:
        """The Sketch's 2D geometry system of geometry/constraint elements."""
        return self._geometry_system

    # Public Functions #
    def get_dependencies(self) -> list[PancadThing]:
        dependencies = set(super().get_dependencies())
        dependencies.update(
            {dep for dep in self.geometry_system.get_dependencies()
             if dep.uid != self.uid}
        )
        return list(dependencies)

    def get_support(self) -> AbstractFeature | AbstractGeometry:
        """Returns the features supporting the sketch in space.

        :raises ValueError: When the sketch is not supported or not in a system.
        """
        if not self.system:
            raise ValueError(f"Sketch '{self.name}' is not in a system")
        index = self.system.get_topo_index(self)
        # Constraints placing the feature should have the same topological index as the feature.
        constraints = [c for c in self.system.get_constraints_on(self)
                       if self.system.get_topo_index(c) == index]
        # Check whether it's possible to get the support
        if not constraints:
            msg = f"Sketch '{self.name}' is not supported"
            if self.system.feature:
                msg = msg + f" inside feature '{self.system.feature.name}'"
            raise ValueError(msg)

        if len(constraints) != 1:
            raise NotImplementedError("Multiple constraints placing a sketch is"
                                      f" not yet supported: {constraints}")
        constraint = constraints[0]
        if constraint.type_name == SketchConstraint.ALIGN_AXES:
            # Satisfying a Sketch's AlignAxes constraint is performed by finding the first
            # FeatureSystem inside a constraint's first Feature dependency. Ex: The FeatureSystem
            # of a FeatureContainer is a common one.
            try:
                feature = next(f for f in constraint.get_dependencies()
                               if f != self and isinstance(f, AbstractFeature))
            except StopIteration as exc:
                deps = constraint.get_dependencies()
                raise ValueError(f"No Feature to act as support in: {deps}") from exc
            try:
                support_system = next(g for g in feature.feature_geometry
                                      if isinstance(g, FeatureSystem))
            except StopIteration as exc:
                raise ValueError(f"Support {feature} does not have a system to align to") from exc
            # The xy plane is assumed
            return support_system.coordinate_system.xy_plane
        raise ValueError(f"Unsupported constraint type for placing sketches: {constraint}")

    def is_equal(self, other: Sketch) -> bool:
        return (self.pose.is_equal(other.pose)
                and self.geometry_system.is_equal(other.geometry_system))

    # Python Dunders #
    def __repr__(self) -> str:
        """Returns the short string representation of the sketch"""
        return super().__repr__().format(details=f"'{self.name}'")
