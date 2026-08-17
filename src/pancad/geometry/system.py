"""A module providing classes defining how systems of geometry are managed."""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pancad.abstract import (
    AbstractGeometrySystem, AbstractFeatureSystem, PancadThing,
    AbstractGeometry, AbstractConstraint, AbstractFeature
)
from pancad.constants import ConstraintReference
from pancad.exceptions import SketchGeometryHasConstraintsError
from pancad.geometry.coordinate_system import CoordinateSystem
from pancad.geometry.unique_lists import (
    SketchGeometryList,
    SketchConstraintList,
    SystemFeatureList,
    FeatureConstraintList,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from uuid import UUID
    from typing import Self

    from pancad.geometry.line import Axis
    from pancad.geometry.plane import Plane
    from pancad.geometry.point import Point

class FeatureSystem(AbstractFeatureSystem):
    """A class managing the relationships between features, their internal
    geometry, and constraints between them.

    :param coordinate_system: The coordinate system at the center of the feature system.
    :param features: A sequence of feature elements.
    :param constraints: A sequence of constraints applied to the geometry in the features.
    :param feature: The feature that this feature system is owned by.
    :param uid: The unique id of this feature system. Auto-generated if not provided.
    """
    def __init__(self,
                 coordinate_system: CoordinateSystem | None=None,
                 features: Iterable[AbstractFeature] | None=None,
                 constraints: Iterable[AbstractConstraint] | None=None, *,
                 feature: AbstractFeature | None=None,
                 uid: str | UUID | None=None) -> None:
        # FeatureSystems are always 3D.
        if not coordinate_system:
            coordinate_system = CoordinateSystem((0, 0, 0))
        if coordinate_system.system:
            raise ValueError(f"coordinate_system already has system: {coordinate_system.system}")
        self._coordinate_system = coordinate_system
        # Initialize references and parent class properties
        self.uid = uid
        self._features = SystemFeatureList(self, [])
        self._constraints = FeatureConstraintList(self, [])
        references = {ConstraintReference.CORE: self,
                      ConstraintReference.CS: self.coordinate_system}
        subreferences = [ConstraintReference.ORIGIN,
                         ConstraintReference.X, ConstraintReference.Y,
                         ConstraintReference.Z,
                         ConstraintReference.XY, ConstraintReference.XZ,
                         ConstraintReference.YZ]
        for sub in subreferences:
            references[sub] = self.coordinate_system.get_reference(sub)
        super().__init__(references, system=self, feature=feature)

        # Add features and constraints to system
        if features is None:
            features = []
        if constraints is None:
            constraints = []
        self.features.extend(features)
        self.constraints.extend(constraints)

    @property
    def coordinate_system(self) -> CoordinateSystem:
        """The CoordinateSystem placing the system's geometry. Read-only."""
        return self._coordinate_system

    @property
    def origin(self) -> Point:
        """The origin of the system's coordinate system."""
        return self.coordinate_system.origin

    @property
    def x_axis(self) -> Axis:
        """The x axis of the system's coordinate system."""
        return self.coordinate_system.x_axis

    @property
    def y_axis(self) -> Axis:
        """The y axis of the system's coordinate system."""
        return self.coordinate_system.y_axis

    @property
    def z_axis(self) -> Axis:
        """The z axis of the system's coordinate system."""
        return self.coordinate_system.z_axis

    @property
    def xy_plane(self) -> Plane:
        """The xy plane of the system's coordinate system."""
        return self.coordinate_system.xy_plane

    @property
    def xz_plane(self) -> Plane:
        """The xz plane of the system's coordinate system."""
        return self.coordinate_system.xz_plane

    @property
    def yz_plane(self) -> Plane:
        """The yz plane of the system's coordinate system."""
        return self.coordinate_system.yz_plane

    @property
    def features(self) -> SystemFeatureList:
        """The features inside the system's context."""
        return self._features

    @features.setter
    def features(self, values: Sequence[AbstractFeature]) -> None:
        msg = ("Full FeatureSystem features cannot be set yet: issue #177,"
               f" features: {values}")
        raise NotImplementedError(msg)

    @property
    def constraints(self) -> FeatureConstraintList:
        """The constraints on the features inside the system's context."""
        return self._constraints

    @constraints.setter
    def constraints(self, values: Sequence[AbstractConstraint]) -> None:
        msg = ("Full FeatureSystem constraints cannot be set yet: issue #177,"
               f" features: {values}")
        raise NotImplementedError(msg)

    @property
    def feature(self) -> AbstractFeature | None:
        """The feature that owns this system."""
        return self._feature

    @feature.setter
    def feature(self, value: AbstractFeature | None) -> None:
        self._feature = value
        for child in self.children.values():
            if child is self:
                continue
            child.feature = value
        for constraint in self.constraints:
            constraint.feature = value

    #Public Methods
    def get_dependencies(self) -> list[PancadThing]:
        dependencies = set()
        for feature in self.features:
            dependencies.update(feature.get_dependencies())
        for constraint in self.constraints:
            dependencies.update(constraint.get_dependencies())
        return list(dependencies)

    def get_topo_index(self, value: AbstractFeature | AbstractConstraint) -> int:
        """Returns the index of the value in the system's topological ordering.

        :raises LookupError: When the value is not in the system.
        """
        if value not in self:
            msg = f"Provided value '{value}' is not in system '{self}'"
            raise LookupError(msg)
        # Determine the topological index of the value
        if isinstance(value, AbstractConstraint):
            # Constraint topological indices are the index of its last
            # constrained feature, since the constraint can't exist without all
            # of its features.
            return max(self.features.index(f) for f in value.get_dependencies()
                       if isinstance(f, AbstractFeature))
        return self.features.index(value)

    def get_constraints_on(self, value: AbstractFeature) -> list[AbstractConstraint]:
        """Returns the constraints applied to the value inside the system."""
        constraints = []
        for constraint in self.constraints:
            deps = constraint.get_dependencies()
            if any(dep.uid == value.uid for dep in deps):
                constraints.append(constraint)
        return constraints

    def get_topo_dependencies(self, value: AbstractFeature) -> list[AbstractFeature]:
        """Returns the dependencies of the value from its topological ordering. For example, a
        sketch inside the system would be dependent on the features constraining its pose.
        """
        dependencies: set[AbstractFeature] = set()
        index = self.get_topo_index(value)

        # Find dependencies from constraints
        for constraint in self.get_constraints_on(value):
            dependencies.update(
                dep for dep in constraint.get_dependencies()
                if isinstance(dep, AbstractFeature) and self.get_topo_index(dep) < index
            )
        return list(dependencies)

    def get_dependents(self, element: PancadThing) -> list[PancadThing]:
        """Returns features that depend on the element, accounting for the topological order."""
        if isinstance(element, AbstractGeometry):
            raise NotImplementedError("Getting the system geometry dependents not supported")
        if isinstance(element, AbstractConstraint):
            raise NotImplementedError("Getting the system constraint dependents not supported")
        if element not in self:
            raise LookupError(f"Element '{element}' is not in system '{self}'")
        if isinstance(element, AbstractFeature):
            index = self.get_topo_index(element)
            return [d for d in self.get_direct_dependents(element)
                    if self.get_topo_index(d) > index]
        raise TypeError(f"Unexpected element type: {type(element)}")

    def get_direct_dependents(self, feature: AbstractFeature) -> list[AbstractFeature]:
        """Finds the dependencies of the feature not accounting for topological order."""
        if feature not in self:
            raise LookupError(f"Provided feature '{feature}' is not in system '{self}'")
        # Get features constrained together with the feature.
        dependents = {d for c in self.get_constraints_on(feature) for d in c.get_dependencies()
                      if d != feature and isinstance(d, AbstractFeature)}
        # Check for features directly referencing the feature.
        for other in self.features:
            if other == feature:
                continue
            if any(dep == feature for dep in other.get_dependencies()):
                dependents.add(other)
        return list(dependents)

    def is_equal(self, other: FeatureSystem) -> bool:
        if not self.coordinate_system.is_equal(other.coordinate_system):
            return False
        if len(self.features) != len(other.features):
            return False
        return all(sf.is_equal(of) for sf, of in zip(self.features, other.features))

    def update(self, other: FeatureSystem) -> Self:
        """Updates the coordinate_system of the system to match
        another system. Does not directly modify the geometry inside the sketch.
        """
        self.coordinate_system.update(other.coordinate_system)
        return self

    # Python Dunders
    def __contains__(self, item: object) -> bool:
        if isinstance(item, PancadThing):
            return item in [*self.features, *self.constraints, self.feature]
        if isinstance(item, str):
            return any(item == t.name for t in [*self.features, *self.constraints, self.feature])
        return False

    def __len__(self) -> int:
        return len(self.coordinate_system)

    def __repr__(self) -> str:
        return super().__repr__().format(
            details=f"({len(self.features)}f{len(self.constraints)}c)"
        )


class SketchGeometrySystem(AbstractGeometrySystem):
    """A class managing the relationships between geometry and constraints.
    This class can act as a standalone set of geometry or be contained inside a
    class instance of a feature like Sketch or FeatureContainer.

    :param coordinate_system: The coordinate system at the center of the geometry system.
    :param geometry: A sequence of geometry elements or a sequence of (geometry, bool) tuples.
        When bools are provided, they indicate whether the geometry should be construction/normal.
    :param constraints: The constraints applied to the geometry.
    :param construction: A subset of the geometry to mark as construction. Defaults to an empty
        set, indicating all geometry is non-construction.
    :param feature: The feature that this system is owned by.
    """
    def __init__(self,
                 coordinate_system: CoordinateSystem,
                 geometry: Iterable[AbstractGeometry | Sequence[tuple[AbstractGeometry, bool]]]
                                    | None=None,
                 constraints: Iterable[AbstractConstraint] | None=None, *,
                 feature: AbstractFeature | None=None, uid: str | UUID | None=None) -> None:
        # Initialize system and feature references first
        self.uid = uid
        self._geometry = SketchGeometryList(self, [])
        self._constraints = SketchConstraintList(self, [])
        if (error := coordinate_system.system) is not None:
            raise ValueError("Expected None for coordinate_system.system,"
                             f" got {error}")
        self._coordinate_system = coordinate_system
        references = {ConstraintReference.CORE: self,
                      ConstraintReference.CS: coordinate_system}
        subreferences = [ConstraintReference.ORIGIN,
                         ConstraintReference.X, ConstraintReference.Y]
        if len(coordinate_system) == 3:
            subreferences.extend(
                [
                    ConstraintReference.Z,
                    ConstraintReference.XY,
                    ConstraintReference.XZ,
                    ConstraintReference.YZ,
                ]
            )
        for sub in subreferences:
            references[sub] = coordinate_system.get_reference(sub)
        super().__init__(references)
        self.system = self # GeometrySystems are their own system
        self.feature = feature

        # Add geometry and constraints to system
        if geometry is None:
            geometry = []
        if constraints is None:
            constraints = []

        self._construction = set()

        for value in geometry:
            if isinstance(value, tuple):
                geometry_element, construction = value
            else:
                geometry_element = value
                construction = False
            self.geometry.append(geometry_element)
            if construction:
                self._construction.add(geometry_element.uid)
        self.constraints.extend(constraints)

    # Properties #
    @property
    def coordinate_system(self) -> CoordinateSystem:
        """The CoordinateSystem placing the system's geometry. Read-only."""
        return self._coordinate_system

    @property
    def construction(self) -> list[bool]:
        """A list of construction/non-construction booleans in the same order as their
        corresponding geometry elements.

        :getter: Returns a list of bools corresponding to each geometry index.
        :setter: Sets the construction tuple after checking that it is the same length as the
            geometry tuple.
        :raises ValueError: Raised when the construction/geometry tuples are not the same length.
        """
        return [g.uid in self._construction for g in self.geometry]

    @property
    def geometry(self) -> SketchGeometryList:
        """All geometry internal to the system."""
        return self._geometry
    @geometry.setter
    def geometry(self, values: Sequence[AbstractGeometry]) -> None:
        new_uids = {geometry.uid for geometry in values}
        old_uids = {geometry.uid for geometry in self.geometry}
        errors = []
        for uid in old_uids - new_uids:
            geometry = self._geometry.get_by_uid(uid)
            try:
                del self._geometry[self._geometry.index(geometry)]
            except SketchGeometryHasConstraintsError as err:
                errors.append(err)
        if errors:
            raise ExceptionGroup("Errors while replacing GeometryList",
                                 errors)
        self._geometry = SketchGeometryList(self, values)

    @property
    def constraints(self) -> SketchConstraintList:
        """All constraints internal to the system."""
        return self._constraints

    @constraints.setter
    def constraints(self, values: Sequence[AbstractConstraint]) -> None:
        self._constraints = SketchConstraintList(self, values)

    @property
    def feature(self) -> AbstractFeature | None:
        """The feature that owns this system."""
        return self._feature

    @feature.setter
    def feature(self, value: AbstractFeature | None) -> None:
        self._feature = value
        for child in self.children.values():
            if child is self:
                continue
            child.feature = value
        for geometry in self.geometry:
            geometry.feature = value
        for constraint in self.constraints:
            constraint.feature = value

    @property
    def origin(self) -> Point:
        """The origin of the system's coordinate system."""
        return self.coordinate_system.origin

    @property
    def x_axis(self) -> Axis:
        """The x axis of the system's coordinate system."""
        return self.coordinate_system.x_axis

    @property
    def y_axis(self) -> Axis:
        """The y axis of the system's coordinate system."""
        return self.coordinate_system.y_axis

    # Public Methods
    def get_dependencies(self) -> list[AbstractFeature]:
        """Gets all the features this system depends on."""
        dependencies = set()
        for constraint in self.constraints:
            dependencies.update(constraint.get_dependencies())
        return list(dependencies)

    def get_dependents(self, element: AbstractGeometry | AbstractConstraint | None=None
                       ) -> list[PancadThing]:
        """Gets the dependents in the scope of the system or an element in the system.

        :param element: The element to look for dependents of. All dependents for the system are
            returned when provided None.
        :raises LookupError: When the element is not in the system.
        """
        if element is None:
            # Get all dependents
            dependents = set()
            for value in [*self.geometry, *self.constraints]:
                dependents.update(self.get_dependents(value))
            return list(dependents)
        # Get element filtered dependents
        if element not in self:
            msg = f"Provided element '{element}' is not in system '{self}'"
            raise LookupError(msg)
        dependents = []
        for constraint in self.constraints:
            # Check if any constraints depend on the element.
            if any(constrained.uid == element.uid
                   for constrained in constraint.get_parents()):
                dependents.append(constraint)
        return dependents

    def get_constraints_on(self, geometry: AbstractGeometry) -> list[AbstractConstraint]:
        """Returns the sketch constraints that are applied to the geometry."""
        constraints = []
        for constraint in self.constraints:
            if any(constrained.uid == geometry.uid
                   for constrained in constraint.get_parents()):
                constraints.append(constraint)
        return constraints

    def add_geometry(self, geometry: AbstractGeometry, construction: bool=False) -> None:
        """Adds an already generated geometry element to the sketch.

        :param geometry: A 2D geometry element.
        :param construction: Sets whether the geometry is construction. Defaults to 'False'.
        """
        self._geometry.append(geometry)
        if construction:
            self._construction.add(geometry.uid)

    def add_constraint(self, constraint: AbstractConstraint) -> None:
        """Adds an already generated constraint to the system.

        :param constraint: A constraint referring to geometry that is already in the system.
        :raises LookupError: Raised when the constraint's dependencies are not in the sketch.
        """
        if all(geometry in self for geometry in constraint.get_parents()):
            self.constraints.append(constraint)
        missing = [geometry for geometry in constraint.get_parents()
                   if geometry not in self]
        raise LookupError(f"{repr(constraint)} dependencies missing: {missing}")

    def get_construction_geometry(self) -> list[AbstractGeometry]:
        """Returns the system's construction geometry."""
        return [g for g in self._geometry if g.uid in self._construction]

    def get_non_construction_geometry(self) -> list[AbstractGeometry]:
        """Returns a tuple of the sketch's non-construction geometry."""
        return [g for g in self._geometry if g.uid not in self._construction]

    def is_equal(self, other: SketchGeometrySystem) -> bool:
        if not self.coordinate_system.is_equal(other.coordinate_system):
            return False
        if len(self.geometry) != len(other.geometry):
            return False
        return all(sg.is_equal(og) for sg, og in zip(self.geometry, other.geometry))

    def update(self, other: SketchGeometrySystem) -> Self:
        """Updates the origin, axes, planes and context of the Sketch to match another Sketch.
        Does not directly modify the geometry inside the sketch.
        """
        self.coordinate_system.update(other.coordinate_system)
        return self

    # Python Dunders #
    def __len__(self) -> int:
        return len(self.coordinate_system)

    def __contains__(self, item: object) -> bool:
        if isinstance(item, (AbstractGeometry, AbstractConstraint)):
            return item in self.geometry or item in self.constraints
        if isinstance(item, str):
            return any(item == t.name for t in [*self.geometry, *self.constraints])
        return False

    def __repr__(self) -> str:
        return super().__repr__().format(
            details=f"({len(self._geometry)}g{len(self._constraints)}c)"
        )

class TwoDSketchSystem(SketchGeometrySystem):
    """A 2-dimensional geometry system.

    :param geometry: The geometry to be placed into the sketch.
    :param constraints: The constraints applied to the geometry in the sketch.
    :param feature: The feature the system is inside of.
    :param coordinate_system: Will be initialized at (0, 0) when None.
    """
    def __init__(self,
                 geometry: Iterable[AbstractGeometry | Sequence[AbstractGeometry, bool]]
                           | None=None,
                 constraints: Iterable[AbstractConstraint] | None=None, *,
                 feature: AbstractFeature | None=None,
                 uid: str | UUID | None=None,
                 coordinate_system: CoordinateSystem | None=None) -> None:
        if coordinate_system is None:
            coordinate_system = CoordinateSystem((0, 0))
        if len(coordinate_system) != 2:
            raise ValueError(f"Expected 2D CS, got '{coordinate_system}'")
        super().__init__(coordinate_system, geometry, constraints,
                         feature=feature, uid=uid)

class ThreeDSketchSystem(SketchGeometrySystem):
    """A 3-dimensional geometry system.

    :param geometry: The geometry to be placed into the sketch.
    :param constraints: The constraints applied to the geometry in the sketch.
    :param feature: The feature the system is inside of.
    :param coordinate_system: Will be initialized at (0, 0, 0) when None.
    """
    def __init__(self,
                 geometry: Iterable[AbstractGeometry | Sequence[AbstractGeometry, bool]]
                           | None=None,
                 constraints: Iterable[AbstractConstraint] | None=None, *,
                 feature: AbstractFeature | None=None,
                 uid: str | UUID | None=None,
                 coordinate_system: CoordinateSystem | None=None) -> None:
        if coordinate_system is None:
            coordinate_system = CoordinateSystem((0, 0, 0))
        if len(coordinate_system) != 3:
            raise ValueError(f"Expected 3D CS, got '{coordinate_system}'")
        super().__init__(coordinate_system, geometry, constraints,
                         feature=feature, uid=uid)

    @property
    def z_axis(self) -> Axis:
        """The z axis of the system's coordinate system."""
        return self.coordinate_system.z_axis

    @property
    def xy_plane(self) -> Plane:
        """The xy plane of the system's coordinate system."""
        return self.coordinate_system.xy_plane

    @property
    def xz_plane(self) -> Plane:
        """The xz plane of the system's coordinate system."""
        return self.coordinate_system.xz_plane

    @property
    def yz_plane(self) -> Plane:
        """The yz plane of the system's coordinate system."""
        return self.coordinate_system.yz_plane
