"""A module defining the lists of unique CAD elements used by features and geometry."""
from __future__ import annotations

from abc import ABCMeta
from collections.abc import MutableSequence
from typing import TYPE_CHECKING, overload, Generic, TypeVar

from pancad.abstract import PancadThing, AbstractConstraint, AbstractFeature, AbstractGeometry
from pancad.exceptions import (
    DupeUidError,
    HasDependentsError,
    MissingCADDependencyError,
)

T = TypeVar("T", bound=PancadThing)
GC = TypeVar("GC", bound=AbstractGeometry | AbstractConstraint)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Sequence, Optional
    from uuid import UUID

    from pancad.abstract import AbstractGeometrySystem
    from pancad.geometry.system import SketchGeometrySystem, FeatureSystem

    ListSlice = slice[int | None, int | None, int | None]


class UniqueCADList(MutableSequence[T], Generic[T], metaclass=ABCMeta):
    """A class managing a mutable list of CAD geometry and constraints inside a system.

    :param parent: The geometry system containing this list.
    :param values: elements to initialize the list with.
    :raises DupeUidError: When trying to add multiple of elements with
        the same uid to the list.
    """
    __type_name = "PancadThing"

    def __init__(self,
                 parent: AbstractGeometrySystem,
                 values: Optional[Sequence[T]]=None) -> None:
        self._parent = parent
        self._values: list[T] = []
        if values is not None:
            self.extend(values)

    # Properties
    @property
    def _type_name(self) -> str:
        """Name used in error messages for this list."""
        return self.__type_name

    # Public Methods
    def get_by_uid(self, uid: str | UUID) -> T:
        """Returns a feature with the matching uid.

        :raises LookupError: When no matching uid is found.
        """
        try:
            return next(value for value in self if value.uid == uid)
        except StopIteration as exc:
            raise LookupError(
                f"No {self._type_name} with uid '{uid}' found."
            ) from exc

    def _get_contents(self) -> list[T]:
        """Returns the full list of contents, including any items in specialized
        indices.
        """
        return self._values

    # Private Methods
    def _raise_if_duped_uid(self, value: T) -> None:
        """Raises a DupeUidError if the geometry's uid is already in the
        list. Used when trying to add geometry to the list.
        """
        if value in self:
            msg = f"{self._type_name} {value} uid: {value.uid} already in list."
            raise DupeUidError(msg)

    def _raise_if_has_dependents(self, value: T) -> None:
        """Raises a HasDependentsError if geometry still has
        constraints. Used when trying to delete geometry from list.
        """
        if dependents := self._parent.get_dependents(value):
            msg = f"{self._type_name} {value} has dependents: {dependents}"
            raise HasDependentsError(msg)

    # Dunders
    @overload
    def __getitem__(self: UniqueCADList[T], index: int) -> T: ...
    @overload
    def __getitem__(self: UniqueCADList[T], index: ListSlice) -> list[T]: ...
    def __getitem__(self: UniqueCADList[T], index: int | ListSlice) -> T | list[T]:
        return self._values[index]

    @overload
    def __delitem__(self, index: int) -> None: ...
    @overload
    def __delitem__(self, index: ListSlice) -> None: ...
    def __delitem__(self, index: int | ListSlice) -> None:
        """Deletes the value from the list after checking deletion validity.

        :raises HasDependentsError: Raised if the feature still has dependents.
        """
        if isinstance(index, slice):
            raise NotImplementedError("Cannot use slices with delitem yet, see #281")
        self._raise_if_has_dependents(self[index])
        del self._values[index]

    def __len__(self) -> int:
        return len(self._values)

    def __contains__(self, value: object) -> bool:
        if not isinstance(value, PancadThing):
            return False
        return any(value.uid == element.uid for element in self._get_contents())

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return str(self._values)


class SystemFeatureList(UniqueCADList[AbstractFeature]):
    """A class managing a mutable list of CAD features inside of a
    FeatureSystem. The list's parent does not contribute to the lists's
    length, but is accessible at index -1.

    :param parent: The system containing this list.
    :param values: Features to initialize the list with.
    """
    __type_name = "Feature"

    def __init__(self,
                 parent: FeatureSystem,
                 values: Sequence[AbstractFeature]) -> None:
        super().__init__(parent, values)

    # Public Methods
    def index(self, value: AbstractFeature, start: int=0, stop: Optional[int]=None) -> int:
        if value is self._parent.feature:
            return -1
        if stop is None:
            return super().index(value, start)
        return super().index(value, start, stop)

    def insert(self, index: int, value: AbstractFeature) -> None:
        """Inserts the object into the list and assigns its system to the
        list's parent.
        """
        self._raise_if_missing_dependencies(value)
        self._raise_if_duped_uid(value)
        self._values.insert(index, value)
        self._assign_system(value)

    def get_by_name(self, name: str) -> AbstractFeature:
        """Returns the first feature with the matching name.

        :raises LookupError: When no matching name is found.
        """
        try:
            return next(value for value in self if value.name == name)
        except StopIteration as exc:
            msg = f"No {self._type_name} with name '{name}' found."
            raise LookupError(msg) from exc

    def _get_contents(self) -> list[AbstractFeature]:
        if self._parent.feature is not None:
            return [self._parent.feature] + super()._get_contents()
        return super()._get_contents()

    def missing_dependencies(self, value: AbstractFeature) -> list[PancadThing]:
        """Returns missing feature dependencies for a feature."""
        return [feature for feature in value.get_dependencies()
                if feature not in self._parent]

    def _raise_if_missing_dependencies(self, value: AbstractFeature) -> None:
        """Raises a MissingCADDependencyError when not all of a constraint's
        dependencies are in the list's system.
        """
        if missing := self.missing_dependencies(value):
            msg = f"{self.__type_name} '{value}' missing dependency: {missing}"
            raise MissingCADDependencyError(msg)

    #Private Methods
    def _assign_system(self, value: AbstractFeature) -> None:
        if value.system is not None:
            raise ValueError(f"{self._type_name} '{value}' is already"
                             f" in another system: '{value.system}'")
        value.system = self._parent

    # Dunders
    @overload
    def __getitem__(self, index: int) -> AbstractFeature: ...
    @overload
    def __getitem__(self, index: ListSlice) -> list[AbstractFeature]: ...
    def __getitem__(self, index: int | ListSlice) -> AbstractFeature | list[AbstractFeature]:
        if index == -1:
            if self._parent.feature:
                return self._parent.feature
            msg = "Cannot get the parent system's feature: {self._parent} not in a feature."
            raise RuntimeError(msg)
        return super().__getitem__(index)

    @overload
    def __setitem__(self, index: int, value: AbstractFeature) -> None: ...
    @overload
    def __setitem__(self, index: slice[int | None, int | None, int | None],
                    value: Iterable[AbstractFeature]) -> None: ...
    def __setitem__(self, index: int | ListSlice,
                    value: AbstractFeature | Iterable[AbstractFeature]) -> None:
        """Replaces object in list and removes the old object's system."""
        if isinstance(index, slice):
            raise NotImplementedError("Cannot use slices with setitem yet, see #281")
        if not isinstance(value, AbstractFeature):
            raise NotImplementedError("Cannot set multiple values with setitem yet, see #281")
        self._raise_if_missing_dependencies(value)
        previous_value = self._values[index] # -1 is not allowed here
        if self[index].uid != value.uid:
            self._raise_if_duped_uid(value)
        self._raise_if_has_dependents(self[index])
        self._values[index] = value
        self._assign_system(value)
        # Remove the system from exiting element
        previous_value.system = None

class FeatureConstraintList(UniqueCADList[AbstractConstraint]):
    """A class managing the mutable list of constraints between features and
    their dependencies.
    """
    __type_name = "Constraint"

    def __init__(self,
                 parent: FeatureSystem,
                 values: Sequence[AbstractConstraint]) -> None:
        for value in values:
            self._raise_if_missing_dependencies(value)
        super().__init__(parent, values)

    # Public Methods
    def insert(self, index: int, value: AbstractConstraint) -> None:
        """Inserts the object into the list and assigns its system to the
        list's parent.

        :raises DupeUidError: When a duped uid value is added to the list.
        :raises MissingCADDependencyError: When not all constraint
            dependencies are in the constraint lists's parent.
        """
        self._raise_if_missing_dependencies(value)
        self._raise_if_duped_uid(value)
        self._values.insert(index, value)
        self._assign_system(value)

    def missing_dependencies(self,
                             value: AbstractConstraint) -> list[AbstractFeature]:
        """Returns missing feature dependencies for a constraint."""
        missing: list[AbstractFeature] = []
        for geometry in value.get_parents():
            if (geometry.feature
                    and geometry.feature not in self._parent
                    and geometry != self._parent):
                missing.append(geometry.feature)
        return missing

    #Private Methods
    def _assign_system(self, value: AbstractConstraint) -> None:
        if value.system is not None:
            msg = (f"{self._type_name} '{value}' is already"
                   f" in another system: '{value.system}'")
            raise ValueError(msg)
        if value.feature is not None:
            msg = (f"{self._type_name} '{value}' is already"
                   f" in another feature: '{value.feature}'")
            raise ValueError(msg)
        if self._parent.feature is None:
            msg = f"This list's parent is not in a feature, cannot assign the feature of {value}"
            raise RuntimeError(msg)
        value.system = self._parent
        value.feature = self._parent.feature

    def _raise_if_missing_dependencies(self, value: AbstractConstraint) -> None:
        """Raises a MissingCADDependencyError when not all of a constraint's
        dependencies are in the list's system.
        """
        if missing := self.missing_dependencies(value):
            msg = f"Constraint '{value}' missing feature dependency: {missing}"
            raise MissingCADDependencyError(msg)

    # Dunders
    @overload
    def __setitem__(self, index: int, value: AbstractConstraint) -> None: ...
    @overload
    def __setitem__(self, index: ListSlice, value: Iterable[AbstractConstraint]) -> None: ...
    def __setitem__(self, index: int | ListSlice,
                    value: AbstractConstraint | Iterable[AbstractConstraint]) -> None:
        """Replaces object in list and removes the old object's system and
        feature.
        """
        if isinstance(index, slice):
            raise NotImplementedError("Cannot use slices with setitem yet, see #281")
        if not isinstance(value, AbstractConstraint):
            raise NotImplementedError("Cannot set multiple values with setitem yet, see #281")
        previous_value = self._values[index] # -1 is not allowed here
        self._raise_if_missing_dependencies(value)
        if self[index].uid != value.uid:
            self._raise_if_duped_uid(value)
        self._raise_if_has_dependents(self[index])
        self._values[index] = value
        self._assign_system(value)
        # Remove the system from exiting element
        previous_value.system = None
        previous_value.feature = None

class FeatureGeometryList(MutableSequence[AbstractGeometry]):
    """A class managing the list of geometry that a feature owns. Feature geometry includes any
    geometry that would need to be deleted if the feature was deleted. This list differs from a
    geometry or constraint list because it is directly owned by a feature rather than existing
    inside a system.
    """
    __type_name = "Feature Geometry"

    def __init__(self, parent: AbstractFeature,
                 values: Optional[Sequence[AbstractGeometry]]=None) -> None:
        self._parent = parent
        self._values: list[AbstractGeometry] = []
        if values is not None:
            self.extend(values)

    def get_by_uid(self, uid: str | UUID) -> PancadThing:
        """Returns a geometry element from the list with the matching uid.

        :raises LookupError: When no matching uid is found.
        """
        try:
            return next(value for value in self if value.uid == uid)
        except StopIteration as exc:
            raise LookupError(
                f"No {self._type_name} with uid '{uid}' found."
            ) from exc

    def insert(self, index: int, value: AbstractGeometry) -> None:
        """Inserts the object into the list and assigns its feature to the
        list's parent.

        :raises DupeUidError: When a duped uid value is added to the list.
        """
        self._raise_if_duped_uid(value)
        self._values.insert(index, value)
        self._assign_feature(value)

    @property
    def _type_name(self) -> str:
        """Name used in error messages for this list."""
        return self.__type_name

    def _raise_if_duped_uid(self, value: PancadThing) -> None:
        """Raises a DupeUidError if the geometry's uid is already in the list. Used when trying to
        add geometry to the list.
        """
        if value in self:
            msg = f"{self._type_name} {value} uid: {value.uid} already in list."
            raise DupeUidError(msg)

    def _assign_feature(self, value: AbstractGeometry) -> None:
        if value.feature is not None:
            raise ValueError(f"{self._type_name} '{value}' is already"
                             f" in another feature: '{value.feature}'")
        value.feature = self._parent

    @overload
    def __getitem__(self, index: int) -> AbstractGeometry: ...
    @overload
    def __getitem__(self, index: ListSlice) -> list[AbstractGeometry]: ...
    def __getitem__(self, index: int | ListSlice) -> AbstractGeometry | list[AbstractGeometry]:
        return self._values[index]

    @overload
    def __delitem__(self, index: int) -> None: ...
    @overload
    def __delitem__(self, index: ListSlice) -> None: ...
    def __delitem__(self, index: int | ListSlice) -> None:
        """Deletes object from list and removes its feature."""
        if isinstance(index, slice):
            raise NotImplementedError("Cannot use slices with setitem yet, see #281")
        previous_value = self._values[index]
        del self._values[index]
        # Remove the feature from exiting geometry
        previous_value.feature = None

    @overload
    def __setitem__(self, index: int, value: AbstractGeometry) -> None: ...
    @overload
    def __setitem__(self, index: ListSlice, value: Iterable[AbstractGeometry]) -> None: ...
    def __setitem__(self, index: int | ListSlice,
                    value: AbstractGeometry | Iterable[AbstractGeometry]) -> None:
        """Replaces object in list and removes the old object's feature."""
        if isinstance(index, slice):
            raise NotImplementedError("Cannot use slices with setitem yet, see #281")
        if not isinstance(value, AbstractGeometry):
            raise NotImplementedError("Cannot set multiple values with setitem yet, see #281")
        previous_value = self._values[index]
        if self[index].uid != value.uid:
            self._raise_if_duped_uid(value)
        self._values[index] = value
        self._assign_feature(value)
        # Remove the feature from exiting geometry
        previous_value.feature = None

    def __contains__(self, value: object) -> bool:
        if not isinstance(value, PancadThing):
            return False
        return any(value.uid == element.uid for element in self._values)

    def __len__(self) -> int:
        return len(self._values)

class UniqueSketchElementList(UniqueCADList[GC], Generic[GC], metaclass=ABCMeta):
    """A class managing the interfaces for CAD sketch lists."""

    # Private Methods
    def _assign_system(self, value: AbstractConstraint | AbstractGeometry) -> None:
        """Assigns the system of the sketch element to the list's parent and the
        feature to the parent's feature.
        """
        if value.system is not None:
            raise ValueError(f"{self._type_name} '{value}' is already"
                             f" in another system: '{value.system}'")
        value.system = self._parent
        if self._parent.feature is not None:
            value.feature = self._parent.feature

    # Dunders
    @overload
    def __delitem__(self, index: int) -> None: ...
    @overload
    def __delitem__(self, index: ListSlice) -> None: ...
    def __delitem__(self, index: int | ListSlice) -> None:
        """Deletes object from list and removes its system."""
        if isinstance(index, slice):
            raise NotImplementedError("Cannot use slices with delitem yet, see #281")
        previous_value = self._values[index] # -1 is not allowed here
        super().__delitem__(index)
        # Remove the system from exiting element
        previous_value.system = None
        previous_value.feature = None

class SketchGeometryList(UniqueSketchElementList[AbstractGeometry]):
    """A class managing a mutable list of geometry. The list's parent does not
    contribute to the lists's length, but is accessible at index -1.

    :param parent: The SketchGeometrySystem that contains this list.
    :param values: Geometry to initialize the list with.
    :raises DupeUidError: When trying to add multiple geometries with the same
        uid to the list.
    """
    __type_name = "Geometry"

    def __init__(self,
                 parent: SketchGeometrySystem,
                 values: Sequence[AbstractGeometry]) -> None:
        super().__init__(parent, values)

    def _get_contents(self) -> list[AbstractGeometry]:
        return [self._parent] + super()._get_contents()

    def insert(self, index: int, value: AbstractGeometry) -> None:
        """Inserts the object into the list and assigns its system to the
        list's parent.

        :raises DupeUidError: When a duped uid value is added to the list.
        """
        self._raise_if_duped_uid(value)
        self._values.insert(index, value)
        self._assign_system(value)

    @overload
    def __getitem__(self, index: int) -> AbstractGeometry: ...
    @overload
    def __getitem__(self, index: ListSlice) -> list[AbstractGeometry]: ...
    def __getitem__(self, index: int | ListSlice) -> AbstractGeometry | list[AbstractGeometry]:
        if index == -1:
            return self._parent
        return super().__getitem__(index)

    @overload
    def __setitem__(self, index: int, value: AbstractGeometry) -> None: ...
    @overload
    def __setitem__(self, index: ListSlice, value: Iterable[AbstractGeometry]) -> None: ...
    def __setitem__(self, index: int | ListSlice,
                    value: AbstractGeometry | Iterable[AbstractGeometry]) -> None:
        """Replaces object in list and removes the old object's system."""
        if isinstance(index, slice):
            raise NotImplementedError("Cannot use slices with setitem yet, see #281")
        if not isinstance(value, AbstractGeometry):
            raise NotImplementedError("Cannot set multiple values with setitem yet, see #281")
        previous_value = self._values[index] # -1 is not allowed here
        if self[index].uid != value.uid:
            self._raise_if_duped_uid(value)
        self._raise_if_has_dependents(self[index])
        self._values[index] = value
        self._assign_system(value)
        # Remove the system from exiting element
        previous_value.system = None
        previous_value.feature = None


class SketchConstraintList(UniqueSketchElementList[AbstractConstraint]):
    """A class managing a mutable list of sketch constraints and their
    dependencies.

    :param parent: The SketchGeometrySystem that contains this list.
    :param values: Constraints to initialize the list with.
    :raises DupeUidError: When trying to add multiple constraints with the same
        uid to the list.
    """
    __type_name = "Constraint"

    def __init__(self,
                 parent: SketchGeometrySystem,
                 values: Sequence[AbstractConstraint]) -> None:
        for value in values:
            self._raise_if_missing_dependencies(value)
        super().__init__(parent, values)

    # Public Methods
    def insert(self, index: int, value: AbstractConstraint) -> None:
        """Inserts the constraint into the constraint list.

        :raises MissingCADDependencyError: When not all constraint
            dependencies are in the constraint lists's associated system.
        :raises DupeUidError: When a duped uid value is added to the list.
        """
        self._raise_if_missing_dependencies(value)
        self._raise_if_duped_uid(value)
        self._values.insert(index, value)
        self._assign_system(value)

    def missing_dependencies(self, value: AbstractConstraint) -> list[AbstractGeometry]:
        """Returns missing geometry dependencies for a constraint."""
        return [geometry for geometry in value.get_geometry() if geometry not in self._parent]

    # Private Methods
    def _raise_if_missing_dependencies(self, value: AbstractConstraint) -> None:
        """Raises a MissingCADDependencyError when not all of a constraint's
        dependencies are in the list's system.
        """
        if missing := self.missing_dependencies(value):
            raise MissingCADDependencyError(f"'{value}' missing: {missing}")

    # Dunders
    @overload
    def __setitem__(self, index: int, value: AbstractConstraint) -> None: ...
    @overload
    def __setitem__(self, index: ListSlice, value: Iterable[AbstractConstraint]) -> None: ...
    def __setitem__(self, index: int | ListSlice,
                    value: AbstractConstraint | Iterable[AbstractConstraint]) -> None:
        """Sets the index to the constraint.

        :raises MissingCADDependencyError: When not all constraint
            dependencies are in the constraint lists's parent.
        """
        if isinstance(index, slice):
            raise NotImplementedError("Cannot use slices with setitem yet, see #281")
        if not isinstance(value, AbstractConstraint):
            raise NotImplementedError("Cannot set multiple values with setitem yet, see #281")
        self._raise_if_missing_dependencies(value)
        previous_value = self._values[index] # -1 is not allowed here
        if self[index].uid != value.uid:
            self._raise_if_duped_uid(value)
        self._raise_if_has_dependents(self[index])
        self._values[index] = value
        self._assign_system(value)
        # Remove the system from exiting element
        previous_value.system = None
        previous_value.feature = None

    def __len__(self) -> int:
        return len(self._values)

    def __contains__(self, value: object) -> bool:
        if isinstance(value, AbstractConstraint):
            return any(value.uid == element.uid for element in self)
        return False
