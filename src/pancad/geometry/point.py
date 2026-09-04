"""A module providing a class to represent points in all CAD programs,
graphics, and other geometry use cases.
"""
from __future__ import annotations

from collections.abc import Collection
import math
from sqlite3 import PrepareProtocol
from typing import TYPE_CHECKING, overload

import numpy as np

from pancad.abstract import AbstractGeometry
from pancad.constants import ConstraintReference
from pancad.utils import trigonometry as trig
from pancad.utils.geometry import parse_vector

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Optional, Type, Self, Literal
    from uuid import UUID

    import numpy.typing as npt

    from pancad.utils.pancad_types import (
        Numpy1D,
        Numpy2D,
        PolarVector,
        SphericalVector,
        SpaceVector,
        Space2DVector,
        Space3DVector,
    )


class Point(AbstractGeometry):
    """A class representing points in 2D and 3D space. Point can return its
    position in cartesian, spherical, or polar coordinates. Point's __init__
    function can only take cartesian coordinates, so either use one of its class
    functions or initialize it with no arguments and modify one of its
    coordinate system specific properties if another coordinate system is
    desired.

    :param components: Either the (x, y) or (x, y, z) as individual Real number
        arguments or as a single vector.
    :param uid: The unique ID of the point for interoperable CAD identification.
    """
    def __init__(self,
                 *components: float | Collection[float],
                 uid: Optional[str | UUID]=None,
                 name: str | None=None):
        self.uid = uid
        self.cartesian = parse_vector(*components)
        super().__init__({ConstraintReference.CORE: self}, name=name)

    # Class Methods #
    @classmethod
    def from_polar(cls, *components: float | Collection[float],
                   uid: Optional[str | UUID]=None) -> Point:
        """Initializes a point from polar coordinates.

        :param components: The (Radius (r), Azimuth (phi)) individual number arguments or as a
            single vector. Azimuth must be in radians.
        :param uid: The unique ID of the point for interoperable CAD
            identification.
        :raises ValueError: When provided a vector not 2 long.
        """
        vector = parse_vector(*components)
        if len(vector) != 2:
            raise ValueError(f"Expected a vector 2 long, got {vector}")
        return cls(trig.polar_to_cartesian(vector), uid=uid)

    @classmethod
    def from_spherical(cls, *components: float | Collection[float],
                       uid: Optional[str | UUID]=None) -> Point:
        """Initializes a point from spherical coordinates (Radius (r), Azimuth
        (phi), Elevation (theta)). Azimuth and Elevation angles must be in
        radians.

        :param components: The (Radius (r), Azimuth (phi), Elevation (theta))
            individual number arguments or as a single vector. Azimuth and
            Elevation must be in radians.
        :param uid: The unique ID of the point for interoperable CAD
            identification.
        :raises ValueError: When provided a vector not 3 long.
        """
        vector = parse_vector(*components)
        if len(vector) != 3:
            raise ValueError(f"Expected a vector 3 long, got {vector}")
        return cls(trig.spherical_to_cartesian(vector), uid=uid)

    # Cartesian Coordinates #
    @property
    def cartesian(self) -> SpaceVector:
        """The cartesian coordinates (x, y) or(x, y, z) of the point.

        :raises ValueError: When provided a sequence not 2 or 3 long.
        """
        return self._cartesian

    @cartesian.setter
    def cartesian(self, value: Collection[float]) -> None:
        vector = trig.to_1d_tuple(value)
        self._cartesian = vector

    @property
    def x(self) -> float:
        """The point's cartesian x-coordinate."""
        return self.cartesian[0]

    @x.setter
    def x(self, value: float) -> None:
        if len(self._cartesian) == 2:
            self.cartesian = (value, self._cartesian[1])
        else:
            self.cartesian = (value, self._cartesian[1], self._cartesian[2])

    @property
    def y(self) -> float:
        """The point's cartesian y-coordinate."""
        return self.cartesian[1]

    @y.setter
    def y(self, value: float) -> None:
        if len(self._cartesian) == 2:
            self.cartesian = (self._cartesian[0], value)
        else:
            self.cartesian = (self._cartesian[0], value, self._cartesian[2])

    @property
    def z(self) -> float:
        """The point's cartesian z-coordinate. Turns a 2D point 3D if set.

        :raises IndexError: When the getter is called on a 2D point.
        """
        if len(self.cartesian) == 3:
            return self.cartesian[2]
        raise IndexError("Cannot get the z value of a 2D point")

    @z.setter
    def z(self, value: float) -> None:
        self.cartesian = (self.cartesian[0], self.cartesian[1], value)

    # Polar/Spherical Coordinates
    @property
    def polar(self) -> PolarVector:
        """The polar coordinates of the point (r, phi). Azimuth angle is and
        must be in radians.

        :raises ValueError: When called if the point is 3D.
        """
        if len(self.cartesian) == 2:
            return trig.cartesian_to_polar(self.cartesian)
        raise ValueError(f"Cannot get the polar coordinates of a {len(self.cartesian)}D Point")

    @polar.setter
    def polar(self, value: Space2DVector) -> None:
        self.cartesian = trig.polar_to_cartesian(value)

    @property
    def spherical(self) -> SphericalVector:
        """The spherical coordinates (r, phi, theta) of the point. Azimuth
        and inclination angles are and must be in radians.

        :raises ValueError: When called if the point is 2D.
        """
        if len(self.cartesian) == 3:
            return trig.cartesian_to_spherical(self.cartesian)
        raise ValueError(f"Cannot get spherical coordinates of a {len(self.cartesian)}D Point")

    @spherical.setter
    def spherical(self, value: Space3DVector) -> None:
        self.cartesian = trig.spherical_to_cartesian(value)

    @property
    def r(self) -> float:
        """The polar/spherical radial distance coordinate of the point.

        :raises ValueError: If r < 0, r is NaN, r == 0 and phi is NaN, or r != 0
            and phi and theta are NaNs.
        """
        if len(self) == 2:
            return self.polar.r
        return self.spherical.r

    @r.setter
    def r(self, value: float) -> None:
        if value < 0:
            raise ValueError("r cannot be less than zero")
        if math.isnan(value):
            raise ValueError("r cannot be NaN")
        if len(self) == 2: # Polar
            if value == 0:
                self.polar = (0, math.nan)
            elif not math.isnan(self.phi):
                self.polar = (value, self.phi)
            else:
                raise ValueError("r must be 0 if phi is NaN")
        else: # Spherical (or invalid)
            if value == 0:
                self.spherical = (value, math.nan, math.nan)
            elif not math.isnan(self.phi) and not math.isnan(self.theta):
                self.spherical = (value, self.phi, self.theta)
            else:
                raise ValueError("r must be 0 if phi and theta are NaN")

    @property
    def phi(self) -> float:
        """The polar/spherical azimuth angle of the point in radians.

        :raises ValueError: If phi is NaN and r != 0, r == 0 and phi is not NaN,
            or if phi is NaN and theta is not NaN.
        """
        if len(self) == 2:
            return self.polar.phi
        return self.spherical.phi

    @phi.setter
    def phi(self, value: float) -> None:
        if len(self) == 2: # Polar
            if self.polar.r != 0 and math.isnan(value):
                raise ValueError("phi cannot be NaN if r is non-zero")
            if self.polar.r == 0 and not math.isnan(value):
                raise ValueError("phi must be NaN if r is zero")
            self.polar = (self.polar.r, value)
        else:
            # Spherical (or invalid)
            if math.isnan(self.spherical.theta) and not math.isnan(value):
                # r would need to be NaN
                raise ValueError("phi can only be set to NaN if theta is already NaN")
            self.spherical = (self.r, value, self.theta)

    @property
    def theta(self) -> float:
        """The spherical inclination angle of the point in radians.

        :raises ValueError: If phi is NaN and theta is not 0, pi or NaN.
        :raises IndexError: When the getter is called on a 2D point.
        """
        return self.spherical.theta

    @theta.setter
    def theta(self, value: float) -> None:
        if (math.isnan(self.spherical.phi)
                and (value not in [0, math.pi] and not math.isnan(value))):
            raise ValueError("theta must be NaN, 0, or pi if phi is NaN")
        self.spherical = (self.spherical.r, self.spherical.phi, value)

    # Public Methods #
    def copy(self) -> Point:
        """Returns a copy of the Point at the same position, different uid."""
        return Point(self.cartesian)

    def is_equal(self, other: Point) -> bool:
        """Returns whether the other geometry is geometrically equal. This is a
        separate check from whether a geometry element is equal to this
        geometry element since the uids would not be the same.
        """
        return np.allclose(self.cartesian, other.cartesian)

    def phi_degrees(self) -> float:
        """Returns the polar/spherical azimuth angle of the Point in degrees."""
        if len(self) == 2:
            return math.degrees(self.polar.phi)
        return math.degrees(self.spherical.phi)

    def theta_degrees(self) -> float:
        """Returns the spherical inclination angle of the point in degrees."""
        return math.degrees(self.spherical.theta)

    def set_phi_degrees(self, value: float) -> Self:
        """Sets the polar/spherical azimuth coordinate of the point using a
        value in degrees. Returns the updated point.
        """
        if len(self) == 2:
            self.polar = (self.polar.r, math.radians(value))
        else:
            self.spherical = (self.spherical.r, math.radians(value), self.spherical.theta)
        return self

    def set_theta_degrees(self, value: float) -> Self:
        """Sets the spherical inclination coordinate of the point using a
        value in degrees. Returns the updated point.
        """
        self.spherical = (self.spherical.r, self.spherical.phi, math.radians(value))
        return self

    def update(self, other: Point) -> Self:
        """Updates the point to match the position of another point.

        :raises ValueError: When trying to update a 2D point to 3D point vice-versa.
        """
        if len(self) != len(other):
            msg = f"Cannot update a {len(self)}D point to a {len(other)}D point"
            raise ValueError(msg)
        self.cartesian = other.cartesian
        return self

    @overload
    def vector(self) -> Numpy1D: ...
    @overload
    def vector(self, vertical: Literal[False]) -> Numpy1D: ...
    @overload
    def vector(self, vertical: Literal[True]) -> Numpy2D: ...
    @overload
    def vector(self, vertical: bool) -> Numpy1D | Numpy2D: ...
    def vector(self, vertical: bool=True) -> Numpy1D | Numpy2D:
        """Returns a numpy vector of the point's cartesian.

        :param vertical: Sets whether to return a vertical vector. Defaults True.
        :returns: The cartesian position vector of the point.
        """
        if vertical:
            array_2d: Numpy2D = np.array(self).reshape(len(self.cartesian), 1)
            return array_2d
        array_1d: Numpy1D = np.array(self)
        return array_1d

    # Python Dunders #
    def __add__(self, other: Collection[float]) -> Numpy1D:
        """Returns the addition of the point's cartesian position vector and another vector of
        the same length as a numpy array. Points can also be added to numpy arrays, but numpy's
        dunders handle those cases through the array dunder.

        :raises ValueError: When vector lengths are unequal or the other is not 2D/3D.
        """
        if isinstance(other, Collection):
            other_array = np.array(other, dtype=np.float64)
            trig.check_vector_shape(other_array.shape)
            try:
                return np.array(self) + other_array
            except ValueError as exc:
                if len(self) != len(other):
                    msg = f"Cannot add {len(self)}D elements to/from {len(other)}D elements"
                    raise ValueError(msg) from exc
                raise
        return NotImplemented

    def __radd__(self, other: Collection[float]) -> Numpy1D:
        return self.__add__(other)

    def __conform__(self, protocol: Type[PrepareProtocol]) -> str:
        """Conforms the point's values for storage in sqlite."""
        if protocol is PrepareProtocol:
            return ";".join(map(str, self.cartesian))
        raise TypeError(f"Expected sqlite3.PrepareProtocol, got {protocol}")

    def __sub__(self, other: Collection[float]) -> Numpy1D:
        """Returns the subtraction of two point's cartesian position vectors as a numpy array."""
        if isinstance(other, Collection):
            other_array = np.array(other, dtype=np.float64)
            trig.check_vector_shape(other_array.shape)
            try:
                return np.array(self) - other_array
            except ValueError as exc:
                if len(self) != len(other):
                    msg = f"Cannot add {len(self)}D elements to/from {len(other)}D elements"
                    raise ValueError(msg) from exc
                raise
        return NotImplemented

    def __rsub__(self, other: Collection[float]) -> Numpy1D:
        if isinstance(other, Collection):
            other_array = np.array(other, dtype=np.float64)
            trig.check_vector_shape(other_array.shape)
            try:
                return other_array - np.array(self)
            except ValueError as exc:
                if len(self) != len(other):
                    msg = f"Cannot add {len(self)}D elements to/from {len(other)}D elements"
                    raise ValueError(msg) from exc
                raise
        return NotImplemented

    def __copy__(self) -> Point:
        """Returns a copy of the point that has the same coordinates, but no
        assigned uid. Can be used with the python copy module.
        """
        return self.copy()

    def __getitem__(self, item: int) -> float:
        """Returns the cartesian coordinates when subscripted. 0 returns x, 1
        returns y, 2 returns z.
        """
        return self.cartesian[item]

    def __len__(self) -> int:
        """Returns the dimension of the Point, 2 or 3."""
        return len(self.cartesian)

    def __iter__(self) -> Iterator[float]:
        """Iterator function to allow the point's cartesian position to be
        output when the point is fed to a list or tuple like function.
        """
        yield from self._cartesian

    def __contains__(self, value: object) -> bool:
        if isinstance(value, (float, int)):
            return value in self._cartesian
        return False

    def __repr__(self) -> str:
        pt_strs = []
        for component in self.cartesian:
            if np.isclose(component, 0):
                pt_strs.append("0")
            else:
                pt_strs.append(f"{component:g}")
        point_str = ",".join(pt_strs)
        return super().__repr__().format(details=f"({point_str})")

    # NumPy Dunders #
    def __array__(self, dtype: None=None, copy: None=None) -> Numpy1D:
        """Array function to allow the point to be fed into a numpy array
        function and return a horizontal numpy array.

        :raises TypeError: When copy is set to False. copy argument only included for numpy
            compatibility.
        """
        array = np.array(list(self))
        if copy is not None and not copy:
            raise TypeError("pancad Point cannot return the original")
        if dtype:
            return array.astype(dtype)
        return array
