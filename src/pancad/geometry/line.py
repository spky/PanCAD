"""A module providing a class to represent lines in all CAD programs,
graphics, and other geometry use cases. Not to be confused with line
segments, which is part of a line that is the shortest distance between two
points.
"""
from __future__ import annotations

from collections.abc import Sequence
from functools import singledispatchmethod
import math
from sqlite3 import PrepareProtocol
from typing import TYPE_CHECKING

import numpy as np

from pancad.abstract import AbstractGeometry
from pancad.constants import ConstraintReference
from pancad.geometry.point import Point
from pancad.utils import trigonometry as trig
from pancad.utils.geometry import closest_to_origin, get_unique_vector
from pancad.utils.quat import Quat

if TYPE_CHECKING:
    from typing import Self, Optional, Type

    from pancad.utils.pancad_types import (
        SpaceVector, Space3DVector, Space2DVector, Numpy1D, Numpy2D, PolarVector, SphericalVector
    )

class Line(AbstractGeometry):
    """A class representing infinite lines in 2D and 3D space. A Line
    instance can be uniquely identified and compared for equality/inequality
    with other lines by using its direction and reference_point. The
    reference_point is the point on the line closest to the origin and should
    not be changed directly. The 'direction' of the line is defined to be
    unique, see the definition in the :meth:`direction` property.

    :param point: A point on the line.
    :param direction: A vector in the direction of the line.
    :param uid: The unique ID of the line.
    """

    zero_tol = np.sqrt(np.finfo(np.float64).eps) # pylint: disable=no-member
    """Any Line direction vector component smaller than this number will be set to 0."""

    def __init__(self, point: Point, direction: Sequence[float] | Numpy1D | Numpy2D,
                 uid: Optional[str]=None,
                 name: str | None=None) -> None:
        self.uid = uid
        self._point_closest_to_origin = Point([0] * len(point)) # Initialize closest point
        self.direction = direction
        if isinstance(point, tuple):
            msg = ("point cannot be a tuple for Line's init function."
                   "Use Line.from_two_points or provide a Point instead")
            raise ValueError(msg)
        self._point_closest_to_origin = Line._closest_to_origin(point.cartesian,
                                                                self.direction)
        super().__init__({ConstraintReference.CORE: self}, name=name)

    # Class Methods
    @classmethod
    def from_two_points(cls,
                        a: Point | Sequence[float] | Numpy1D,
                        b: Point | Sequence[float] | Numpy1D,
                        uid: Optional[str]=None,
                        name: str | None=None) -> Self:
        """Returns a Line instance defined by points a and b. 2D points will
        produce 2D lines, 3D produce 3D lines, and 2D and 3D points cannot
        be mixed.

        :param a: A pancad Point on the line.
        :param b: A pancad Point on the line that is not the same as point a.
        :param uid: The unique ID of the line.
        :returns: A Line that is coincident with points a and b.
        """
        if not isinstance(a, Point):
            a = Point(a)
        if not isinstance(b, Point):
            b = Point(b)
        if a.is_equal(b):
            raise ValueError("Defining points are at the same position")
        return cls(a, b - a, uid, name=name)

    @classmethod
    def from_slope_and_y_intercept(cls,
                                   slope: float,
                                   intercept: float,
                                   uid: Optional[str]=None,
                                   name: str | None=None) -> Self:
        """Returns a 2D line described by y = mx + b.

        :param slope: The slope (m) of the line.
        :param intercept: The y-intercept (b) of the line.
        :param uid: The unique id of the line.
        :returns: A Line with the provided slope and intercept.
        """
        if slope == 0: # Horizontal
            points = (Point(0, intercept), Point(1, intercept))
        else:
            points = (Point(0, intercept), Point(1, slope + intercept))
        return cls.from_two_points(*points, uid, name=name)

    @classmethod
    def from_point_and_angle(cls,
                             point: Point | Sequence[float] | Numpy1D,
                             phi: float,
                             theta: Optional[float]=None,
                             uid: Optional[str]=None,
                             name: str | None=None) -> Self:
        """Return a line from a given point and phi or phi and theta. The Line
        will be 2D if point is 2D. The Line will be 3D if point is 3D, phi
        is provided, and theta is provided.

        :param point: A point on the line.
        :param phi: The azimuth angle of the line around the point in radians.
        :param theta: The inclination angle of a 3D line around the point in
            radians.
        :returns: A Line that runs through the point in a direction defined by
            the provided angles.
        """
        if not isinstance(point, Point):
            point = Point(point)
        if len(point) == 2:
            if theta is not None:
                raise ValueError("Expected None for theta for a 2D point.")
            direction_end_pt = Point(1, 0)
            direction_end_pt.phi = phi
        else:
            direction_end_pt = Point(1, 0, 0)
            if theta is None:
                raise ValueError("Expected theta for a 3D point.")
            direction_end_pt.spherical = (1, phi, theta)
        return cls(point, tuple(direction_end_pt), uid, name=name)

    @classmethod
    def from_x_intercept(cls, x_intercept: float, uid: Optional[str]=None,
                         name: str | None=None) -> Self:
        """Returns a 2D vertical line that passes through the x intercept.

        :param x_intercept: The value of x where the line crosses the x-axis.
        :param uid: The unique ID of the line.
        :returns: A vertical line coincident with (x_intercept, 0).
        """
        return cls(Point(x_intercept, 0), (0, 1), uid, name=name)

    @classmethod
    def from_y_intercept(cls, y_intercept: float, uid: Optional[str]=None,
                         name: str | None=None) -> Self:
        """Returns a 2D horizontal line that passes through the y intercept.

        :param y_intercept: The value of y where the line crosses the y-axis.
        :param uid: The unique ID of the line.
        :returns: A horizontal line coincident with (0, y_intercept).
        """
        return cls(Point(0, y_intercept), (1, 0), uid, name=name)

    # Properties
    @property
    def direction(self) -> SpaceVector:
        """The unique direction of the line with cartesian components.

        pancad Line Directions in 2D are defined to be unique since infinite
        lines do not have a true direction. For a given vector, the unique
        direction is defined by these rules:

        1. The direction vector must have a magnitude of 1.
        2. The z component must be positive or 0.
        3. If the z component is 0 or the line is 2D, the y component must be
           positive or 0.
        4. If both the y and z components are 0, the x component must be
           positive.

        :getter: Returns the direction of the line.
        :setter: Finds and sets the vector's unique direction vector as the
            direction of the Line. Effectively rotates the line about its point
            closest to the origin.
        """
        return self._direction

    @direction.setter
    def direction(self, vector: Sequence[float] | Numpy1D | Numpy2D) -> None:
        parsed_vector = trig.to_1d_np(vector)
        if not np.any(parsed_vector):
            raise ValueError("Direction vector cannot be zero vector")
        self._direction = self._unique_direction(parsed_vector)
        new_closest = self._closest_to_origin(self._point_closest_to_origin.cartesian,
                                              self.direction)
        self._point_closest_to_origin.update(new_closest)

    @property
    def direction_polar(self) -> PolarVector:
        """The unique direction of the line with polar components.

        :getter: Returns the direction of the line as a (r, phi) tuple. Phi is
            the azimuth angle in radians.
        :setter: Finds and sets the polar vector's unique direction vector as
            the direction of the Line.
        """
        if len(self.direction) == 2:
            return trig.cartesian_to_polar(self.direction)
        raise ValueError("Cannot return the polar direction vector of a 3D line")

    @direction_polar.setter
    def direction_polar(self, vector: Sequence[float] | Numpy1D | Numpy2D) -> None:
        parsed_vector = trig.to_1d_tuple(vector)
        if len(parsed_vector) != 2:
            raise ValueError("Expected a 2 long vector.")
        self.direction = trig.polar_to_cartesian(parsed_vector)

    @property
    def direction_spherical(self) -> SphericalVector:
        """The unique direction of the line with spherical components.

        :getter: Returns the direction of the line as a (r, phi, theta) tuple.
            phi and theta are the azimuth and inclination angles respectively,
            in radians.
        :setter: Finds and sets the spherical vector's unique direction vector
            as the direction of the Line.
        """
        if len(self.direction) == 3:
            return trig.cartesian_to_spherical(self.direction)
        raise ValueError("Cannot return the spherical direction vector of a 2D line.")

    @direction_spherical.setter
    def direction_spherical(self, vector: Sequence[float] | Numpy1D | Numpy2D) -> None:
        parsed_vector = trig.to_1d_tuple(vector)
        if len(parsed_vector) != 3:
            raise ValueError("Expected a 3 long vector.")
        self.direction = trig.spherical_to_cartesian(parsed_vector)

    @property
    def phi(self) -> float:
        """The polar/spherical azimuth component of the line's direction in
        radians.

        :getter: Returns the azimuth component of the line's direction.
        :setter: Read-only.
        """
        return trig.phi_of_cartesian(self.direction)

    @property
    def reference_point(self) -> Point:
        """The point on the line closest to the origin.

        :getter: Returns a copy of the Point closest to the origin on the line.
        :setter: Read-only.
        """
        return self._point_closest_to_origin.copy()

    @property
    def slope(self) -> float:
        """The slope of the line (m in y = mx + b), only available if the line
        is 2D.

        :getter: Returns the slope of the line.
        :setter: Sets the slope of the line while keeping the y intercept (b
            in y = mx + b) the same.
        """
        if len(self) == 2:
            if self.direction[0] == 0:
                return math.nan
            return self.direction[1] / self.direction[0]
        raise ValueError("slope is not defined for a 3D line")

    @property
    def theta(self) -> float:
        """The spherical inclination component of the line's direction in
        radians.

        :getter: Returns the inclination angle of the line's direction
        :setter: Read-only.
        """
        if len(self.direction) == 3:
            return trig.theta_of_cartesian(self.direction)
        raise ValueError("Cannot return the spherical theta of a 2D line")

    @property
    def x_intercept(self) -> float:
        """The x-intercept of the 2D line (x when y = 0 in y = mx + b), raises
        a ValueError if the line is 3D.

        :getter: Returns the x-intercept of the line.
        :setter: Sets the x-intercept of the line while keeping the slope (m
            in y = mx + b) constant.
        """
        if len(self) == 2:
            if self.direction[0] == 1:
                return math.nan
            if self.direction[0] == 0:
                return self.reference_point.x
            return ((self.slope*self.reference_point.x - self.reference_point.y)
                    / self.slope)
        raise ValueError("x-intercept is not defined for a 3D line")

    @property
    def y_intercept(self) -> float:
        """The y-intercept of the line (b in y = mx + b), only available if
        the line is 2D.

        :getter: Returns the y-intercept of the line
        :setter: Sets the y-intercept of the line while keeping the slope (m
            in y = mx + b) constant.
        """
        if len(self) == 2:
            if self.direction[0] == 0:
                return math.nan
            return self.reference_point.y - self.slope*self.reference_point.x
        raise ValueError("y-intercept is not defined for a 3D line")

    # Public Methods
    def copy(self) -> Line:
        """Returns a copy of the Line.

        :returns: A new Line with the same position and direction as this Line.
        """
        return Line(self.reference_point, self.direction)

    def is_equal(self, other: Line) -> bool:
        return (self.reference_point.is_equal(other.reference_point)
                and np.allclose(self.direction, other.direction))

    def get_parametric_point(self, t: float) -> Point:
        """Returns the point at parameter t where a, b, and c are defined by
        the unique unit vector direction of the line and initialized at the
        point closest to the origin.

        :param t: The value of the line parameter.
        :returns: The Point on the line corresponding to the parameter's value.
        """
        return Point(np.array(self.reference_point)
                     + trig.to_1d_np(self.direction)*t)

    def get_parametric_constants(self) -> (tuple[float, float, float, float] |
                                           tuple[float, float, float, float, float, float]):
        """Returns a tuple containing parameters for the line. The reference
        point is used for the initial position and the line's direction vector
        is used for a, b, and c.

        :returns: Line parameters (x0, y0, z0, a, b, c)
        """
        return (*self.reference_point.cartesian, *self.direction)

    def move_to_point(self,
                      point: Point | Sequence[float] | Numpy1D,
                      phi: Optional[float]=None,
                      theta: Optional[float]=None) -> Self:
        """Moves the line to go through a point and changes the line's
        direction's around that point.

        :param point: A point the user wants to be on the line
        :param phi: The line's new azimuth angle around the point in radians. If
            None, the Line's azimuth angle remains constant.
        :param theta: The line's new inclination angle around the point in
            radians. If None, the Line's inclination angle remains constant.
        :returns: The line with an updated reference_point that goes through the
            point.
        """
        if not isinstance(point, Point):
            point = Point(point)
        if theta is not None and len(self) == 2:
            raise ValueError("Theta can only be set on 3D Lines")
        if phi is not None or theta is not None:
            direction_end_pt = Point(self.direction)
            if phi is not None and theta is not None and len(self) == 3:
                direction_end_pt.spherical = (1, phi, theta)
            elif phi is not None and theta is None:
                direction_end_pt.phi = phi
            elif phi is None and theta is not None:
                direction_end_pt.theta = theta
            self.direction = tuple(direction_end_pt)
        new_closest = self._closest_to_origin(point.cartesian, self.direction)
        self._point_closest_to_origin.update(new_closest)
        return self

    def update(self, other: Line) -> Self:
        """Updates the line to match the position and direction of another line.

        :param other: The line to update to.
        :returns: The updated Line.
        """
        self._point_closest_to_origin.update(other.reference_point)
        self.direction = other.direction
        return self

    @staticmethod
    def _closest_to_origin(point: SpaceVector, vector: SpaceVector) -> Point:
        """Returns the Point on the Line closest to the origin."""
        return Point(closest_to_origin(point, vector))

    def _unique_direction(self, vector: SpaceVector | Numpy1D) -> SpaceVector:
        """Returns a unit vector that can uniquely identify the direction of
        the given vector. Does so flipping the unit vector if necessary to
        ensure there can only ever be one vector for every direction.

        :param vector: A 1D vector of cartesian coordinates.
        :returns: The unique unit vector to represent the vector's direction.
        """
        unit_vector = get_unique_vector(trig.get_unit_vector(vector))
        return trig.to_1d_tuple(unit_vector)

    # Python Dunders #
    def __conform__(self, protocol: Type[PrepareProtocol]) -> str:
        if protocol is PrepareProtocol:
            dimensions = [*self.reference_point, *self.direction]
            return ";".join(map(str, dimensions))
        raise TypeError(f"Expected sqlite3.PrepareProtocol, got {protocol}")

    def __copy__(self) -> Line:
        """Returns a copy of the line that has the same closest to origin
        point and direction, but a different uid. Can be used with the python
        copy module.
        """
        return self.copy()

    def __len__(self) -> int:
        """Returns whether the Line is 2D or 3D."""
        return len(self.direction)

    def __repr__(self) -> str:
        """Returns the short string representation of the line."""
        pt_strs, direction_strs = [], []
        for i, component in enumerate(self.direction):
            if np.isclose(self._point_closest_to_origin[i], 0):
                pt_strs.append("0")
            else:
                pt_strs.append(f"{self._point_closest_to_origin[i]:g}")

            if np.isclose(component, 0):
                direction_strs.append("0")
            else:
                direction_strs.append(f"{component:g}")
        point_str = ",".join(pt_strs)
        direction_str = ",".join(direction_strs)
        return super().__repr__().format(
            details=f"({point_str})d({direction_str})"
        )

class Axis(AbstractGeometry):
    """A class representing infinite lines with direction in 2D and 3D space.

    :param point: A point on the axis.
    :param direction: A vector in the direction of the axis.
    :param uid: The unique ID of the axis.
    """

    def __init__(self, point: Point | Sequence[float] | Numpy1D,
                 direction: Sequence[float] | Numpy1D | Numpy2D,
                 uid: Optional[str]=None) -> None:
        self.uid = uid
        if not isinstance(point, Point):
            point = Point(point)
        self._line = Line(point, direction)
        self.direction = direction
        super().__init__({ConstraintReference.CORE: self})

    # Properties
    @property
    def direction(self) -> SpaceVector:
        """The direction of the axis with cartesian components.

        :getter: Returns the direction of the line.
        :setter: Sets the axis direction vector, effectively rotating the axis
            about its point closest to the origin.
        :raises ValueError: When the direction vector is a zero vector or does
            not match the dimension of the Axis.
        """
        return self._direction

    @direction.setter
    def direction(self, vector: Sequence[float] | Numpy1D | Numpy2D) -> None:
        parsed_vector = trig.to_1d_np(vector)
        if not np.any(parsed_vector):
            raise ValueError("Direction vector cannot be zero vector")
        if len(parsed_vector) != len(self._line):
            raise ValueError("Direction vector must be the same dimension as the Axis")
        self._direction = trig.to_1d_tuple(trig.get_unit_vector(parsed_vector))
        # Axis uses Line to inform geometry, but the Line shouldn't be referenced
        # by constraints. Axis should be referenced directly.
        self._line.direction = parsed_vector

    @property
    def reference_point(self) -> Point:
        """The point on the axis closest to the origin.

        :getter: Returns a copy of the Point closest to the origin on the axis.
        :setter: Read-only.
        """
        return self._line.reference_point

    @property
    def reference_line(self) -> Line:
        """Returns the Line that is coincident with the Axis.

        :getter: Returns a copy of the Line coincident with the Axis.
        :setter: Read-only.
        """
        return self._line.copy()

    # Public Methods
    def copy(self) -> Axis:
        """Returns a copy of the Axis.

        :returns: A new Axis with the same position and direction as this Axis.
        """
        return Axis(self.reference_point, self.direction)

    def is_equal(self, other: Axis) -> bool:
        """Returns whether the other geometry is geometrically equal. This is a
        separate check from whether a geometry element is equal to this
        geometry element since the uids would not be the same.
        """
        return (np.allclose(self.direction, other.direction)
                and self.reference_point.is_equal(other.reference_point))

    def move_to_point(self,
                      point: Point | Sequence[float] | Numpy1D,
                      direction: Optional[Sequence[float] | Numpy1D | Numpy2D]=None) -> Self:
        """Moves the axis to go through the point. Leaves direction constant
        unless provided.

        :param point: A point the Axis should go through.
        :param direction: The new direction the Axis should point. Defaults to
            leaving the direction constant.
        :returns: The updated Axis to enable chaining.
        """
        self._line.move_to_point(point)
        if direction is not None:
            self.direction = direction
            self._line.direction = direction
        return self

    def update(self, other: Axis) -> Self:
        """Updates the Axis to match the position and direction of another Axis.

        :param other: The Axis to update to.
        :returns: The updated Axis to enable chaining.
        """
        self.direction = other.direction
        self._line.update(other.reference_line)
        return self

    @singledispatchmethod
    def rotate(self, rotation: Numpy2D | Quat) -> Self:
        """Rotates the axis about its point closest to the origin.

        :param rotation: The matrix or quaternion to rotate with.
        :returns: The updated Axis to enable chaining.
        :raises ValueError: When provided a rotation that does not correspond to
            the dimensions of the Axis.
        """
        raise TypeError(f"Expected numpy array or Quat, got: {rotation}")

    @rotate.register(Quat)
    def _with_quaternion(self, rotation: Quat) -> Self:
        try:
            self.direction = rotation.rotate(self.direction)
        except ValueError as exc:
            if len(self) == 2:
                msg = f"Cannot rotate 2D Axis with Quat: {rotation}"
                exc.add_note(msg)
            raise
        self._line.direction = self.direction
        return self

    @rotate.register(np.ndarray)
    def _with_matrix(self, rotation: Numpy2D) -> Self:
        try:
            new = rotation @ self.direction
        except ValueError as exc:
            if "has a mismatch in its core dimension" in str(exc):
                shape = "x".join(map(str, rotation.shape))
                msg = (f"Cannot rotate {len(self)}D Axis"
                       f" with {shape} rotation matrix: \n{rotation}")
                exc.add_note(msg)
            raise
        self.direction = trig.to_1d_tuple(new + 0)
        self._line.direction = self.direction
        return self

    # Dunders
    def __len__(self) -> int:
        """Returns whether the Axis is 2D or 3D."""
        return len(self._line)

    def __repr__(self) -> str:
        direction_strs = []
        for component in self.direction:
            if np.isclose(component, 0):
                direction_strs.append("0")
            else:
                direction_strs.append(f"{component:g}")
        pt_strs = []
        for component in self.reference_point:
            if np.isclose(component, 0):
                pt_strs.append("0")
            else:
                pt_strs.append(f"{component:g}")
        direction_str = ",".join(direction_strs)
        pt_str = ",".join(pt_strs)
        return super().__repr__().format(details=f"({pt_str})({direction_str})")
