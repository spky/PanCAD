"""A module providing classes for horizontal, vertical, and direct distance constraints. Distance
can be used to force geometry elements to be at set distances from specified portions of each
other. The horizontal and vertical variants can only be applied to 2D geometry and allow the
distance to be limited to the x or y direction respectively.
"""
from __future__ import annotations

import math
from abc import abstractmethod
from typing import TYPE_CHECKING

from pancad.abstract import AbstractConstraint
from pancad.constants import SketchConstraint

if TYPE_CHECKING:
    from uuid import UUID

    from pancad.abstract import AbstractGeometry, AbstractGeometrySystem

class AbstractValue(AbstractConstraint):
    """An abstract class of constraints that can be applied to one or more geometries that has a
    value associated with it.
    """

    VALUE_STR_FORMAT = "{value}{unit}"

    @property
    @abstractmethod
    def value(self) -> float:
        """The value that the constraint enforces."""

    @property
    def unit(self) -> str | None:
        """The unit of the constraint's value."""
        return self._unit

    @unit.setter
    def unit(self, value: str | None) -> None:
        self._unit = value

    # Public Methods
    def get_value_string(self, include_unit: bool=True) -> str:
        """Returns a string of the value of the constraint.

        :param include_unit: Whether to include the unit in the output. Defaults to 'True'.
        :returns: A string with the constraint's value. Adds the unit if include_unit is 'True'.
        """
        if include_unit:
            return self.VALUE_STR_FORMAT.format(value=self.value, unit=self.unit)
        return str(self.value)

    def __str__(self) -> str:
        super_str = super().__str__().removesuffix(">")
        return f"{super_str}[{self.value}{self.unit}]>"

class Angle(AbstractValue):
    """A class representing angle value constraints between lines. Stores and returns angles in
    degrees by default since nearly all CAD programs take user angle inputs in degrees. Can
    constrain:

    - :class:`~pancad.geometry.CoordinateSystem`
    - :class:`~pancad.geometry.Line`
    - :class:`~pancad.geometry.LineSegment`
    - :class:`~pancad.geometry.Ellipse`

    :param geometry: The pair of geometry elements to be constrained.
    :param value: Angle value in degrees or radians.
    :param quadrant: Quadrant selection between 1 (+, +), 2 (-, +), 3 (-, -) and 4 (+, -). Initial
        selection is only dependent on the first element. Example: the first quadrant is the one
        immediately clockwise of the direction vector of the firest element. The clockwise
        selection method is to simulate how users pick quadrants visually.
    :param uid: Unique identifier of the constraint.
    :param is_radians: Whether provided value is in radians. Defaults to False.
    :param system: The geometry system the constraint will be in.
    :raises ValueError: When the geometry elements are not 2D and when not provided 2 elements.
    """

    type_name = SketchConstraint.ANGLE

    def __init__(self,
                 *geometry: AbstractGeometry,
                 value: float,
                 quadrant: int,
                 uid: UUID | str | None=None,
                 is_radians: bool=False,
                 system: AbstractGeometrySystem | None=None,
                 name: str | None=None) -> None:
        if len(geometry) != 2:
            raise ValueError(f"Expected 2 geometries, provided {geometry}")
        if any(len(g) != 2 for g in geometry):
            non_two_dimensional = [g for g in geometry if len(g) != 2]
            raise ValueError(f"Non-2D Geometry provided: {non_two_dimensional}")
        self._geometry = geometry
        self.uid = uid
        super().__init__(system, name=name)
        self.quadrant = quadrant
        self.unit = "degrees"
        if is_radians:
            self.value = math.degrees(value)
        else:
            self.value = value

    @property
    def quadrant(self) -> int:
        """The quadrant that the angle constraint is applied to.

        :raises ValueError: When the quadrant is not 1, 2, 3, or 4.
        """
        return self._quadrant

    @quadrant.setter
    def quadrant(self, value: int) -> None:
        if value not in {1, 2, 3, 4}:
            raise ValueError("Quadrant must be 1, 2, 3, or 4")
        self._quadrant = value

    @property
    def value(self) -> float:
        """Value of the angle constraint in degrees."""
        return self._value

    @value.setter
    def value(self, value: float) -> None:
        self._value = value

    # Public Methods
    def get_value(self, in_radians: bool=False) -> float:
        """Returns the value of the angle constraint.

        :param in_radians: Whether to return the value in radians or degrees. Defaults to 'False'.
        :returns: The angle value in degrees. Returns in radians when 'in_radians' is 'True'.
        """
        if in_radians:
            return math.radians(self.value)
        return self.value

class AbstractDistance(AbstractValue):
    """An abstract class of constraints that can be applied to one or more geometries that has a
    distance associated with it.

    :param geometry: The geometry elements to be constrained.
    :param value: Distance value.
    :param uid: Unique identifier of the constraint. Defaults to None.
    :param unit: The unit of the distance value. Defaults to None.
    :param system: The geometry system the constraint will be in.
    :raises ValueError: When the geometry elements' dimensions do not match.
    """

    def __init__(self, *geometry: AbstractGeometry,
                 value: float, uid: UUID | str | None=None, unit: str | None=None,
                 system: AbstractGeometrySystem | None=None,
                 name: str | None=None) -> None:
        self.uid = uid
        super().__init__(system, name=name)
        if len(set(len(g) for g in geometry)) != 1:
            raise ValueError(f"Geometry not all the same dimension: {geometry}")
        self._geometry = geometry
        self.unit = unit
        self.value = value

    @property
    def value(self) -> float:
        """The distance the constraint enforces. When distance direction can be well-defined, the
        first direction-dependent geometry defines how positive distance is defined. Ex: The
        first plane's normal vector of two constrained planes is the direction of positive
        distance. Distance should be nonnegative when distance direction cannot be well-defined.
        """
        return self._value

    @value.setter
    def value(self, value: float) -> None:
        self._value = value

class Abstract2GeometryDistance(AbstractDistance):
    """An abstract class of constraints that can be applied **exactly two** geometries that has a
    distance associated with it. Distances cannot be negative.

    :param geometry: The pair of geometry elements to be constrained.
    :param value: Distance value.
    :param uid: Unique identifier of the constraint. Defaults to None.
    :param unit: The unit of the distance value. Defaults to None.
    :param system: The geometry system the constraint will be in.
    :raises ValueError: When not provided 2 elements or the elements are not the same dimension.
    """
    def __init__(self, *geometry: AbstractGeometry,
                 value: float, uid: UUID | str | None=None, unit: str | None=None,
                 system: AbstractGeometrySystem | None=None,
                 name: str | None=None) -> None:
        if len(geometry) != 2:
            raise ValueError(f"Expected 2 geometries, provided {geometry}")
        super().__init__(*geometry, value=value, uid=uid, unit=unit, system=system, name=name)

class Abstract1GeometryDistance(AbstractDistance):
    """An abstract class of constraints that can be applied **exactly one**
    geometry that has a distance associated with it. Distances cannot be
    negative.

    :param geometry: The geometry element to be constrained.
    :param value: Distance value.
    :param uid: Unique identifier of the constraint. Defaults to None.
    :param unit: The unit of the distance value. Defaults to None.
    :param system: The geometry system the constraint will be in.
    """
    def __init__(self, *geometry: AbstractGeometry,
                 value: float, uid: UUID | str | None=None, unit: str | None=None,
                 system: AbstractGeometrySystem | None=None,
                 name: str | None=None) -> None:
        if len(geometry) != 1:
            raise ValueError(f"Expected 2 geometries, provided {geometry}")
        super().__init__(*geometry, value=value, uid=uid, unit=unit, system=system, name=name)

# 2D and 3D Distance Classes #
################################################################################

class Distance(Abstract2GeometryDistance):
    """A constraint that defines the direct distance between two elements in 2D
    or 3D.
    """

    type_name = SketchConstraint.DISTANCE

# 2D Only Classes #
################################################################################
class AbstractDistance2D(Abstract2GeometryDistance):
    """An abstract class for 2D distance constraints."""
    def __init__(self, *geometry: AbstractGeometry,
                 value: float, uid: UUID | str | None=None, unit: str | None=None,
                 system: AbstractGeometrySystem | None=None,
                 name: str | None=None) -> None:
        if any(len(g) != 2 for g in geometry):
            non_two_dimensional = [g for g in geometry if len(g) != 2]
            raise ValueError(f"Non-2D geometry provided: {non_two_dimensional}")
        super().__init__(*geometry, value=value, uid=uid, unit=unit, system=system, name=name)

class HorizontalDistance(AbstractDistance2D):
    """A constraint that sets the horizontal distance between two elements."""

    type_name = SketchConstraint.DISTANCE_HORIZONTAL

class VerticalDistance(AbstractDistance2D):
    """A constraint that sets the vertical distance between two elements."""

    type_name = SketchConstraint.DISTANCE_VERTICAL

class Radius(Abstract1GeometryDistance):
    """A constraint that sets the radius of a curve."""

    type_name = SketchConstraint.DISTANCE_RADIUS

class Diameter(Abstract1GeometryDistance):
    """A constraint that sets the diameter of a curve."""

    type_name = SketchConstraint.DISTANCE_DIAMETER
