"""A module providing a class to represent line segments in all CAD programs,
graphics, and other geometry use cases.
"""
from __future__ import annotations

from sqlite3 import PrepareProtocol
from typing import TYPE_CHECKING

import numpy as np

from pancad.abstract import AbstractGeometry
from pancad.constants import ConstraintReference as CR
from pancad.geometry.point import Point
from pancad.geometry.line import Line
from pancad.utils import trigonometry as trig
from pancad.utils.geometry import parse_vector

if TYPE_CHECKING:
    from collections.abc import Collection
    from typing import Self, Type

    from pancad.utils.pancad_types import SpaceVector


class LineSegment(AbstractGeometry):
    """A class representing finite lines in 2D and 3D space.

    :param start: The start point of the line segment.
    :param end: The end point of the line segment.
    :param uid: The unique id of the line segment.
    :param name: The user defined name of the line segment.
    """
    def __init__(self, start: Collection[float], end: Collection[float], *,
                 uid: str | None=None,
                 name: str | None=None) -> None:
        self._start = Point(start)
        self._end = Point(end)
        if self.start.is_equal(self.end):
            raise ValueError(f"start/end cannot be at the same location. Got: {start} and {end}")
        self.uid = uid
        children = {CR.CORE: self, CR.START: self.start, CR.END: self.end}
        super().__init__(children, name=name)

    # Class Methods
    @classmethod
    def from_point_length_angle(cls,
                                start: Collection[float],
                                *components: float | Collection[float],
                                uid: str | None=None,
                                name: str | None=None) -> Self:
        """Returns a LineSegment defined by a point and a length, azimuth angle
        phi, and inclination angle theta relative to the point.

        :param point: A Point, or iterable with 2 or 3 dimensions.
        :param components: The length (r), azimuth angle (phi), and (for 3D
            only) the inclination angle (theta) of the vector from the start
            point to the end point. A polar/spherical vector. Angles must be in
            radians.
        :returns: A line segment with its start at point, and end at the point's
            position plus the polar/spherical vector.
        :raises TypeError:  When provided a single component that is not Sequence
            or when 2 or more non-Real arguments.
        :raises ValueError: When provided too many components or a start point
            and components with differing dimensions.
        """
        vector = parse_vector(*components)
        cartesian: SpaceVector
        if len(vector) == 2:
            cartesian = trig.polar_to_cartesian(vector)
        else:
            cartesian = trig.spherical_to_cartesian(vector)
        try:
            end = np.array(start) + cartesian
        except ValueError as exc:
            if len(start) != len(cartesian):
                msg = f"start/components dimension mismatch: {start}, {components}"
                raise ValueError(msg) from exc
            raise
        return cls(start, end, uid=uid, name=name)

    # Properties
    @property
    def direction(self) -> SpaceVector:
        """The direction of the line segment defined as the unit vector pointing
        from start to end with cartesian components.

        The direction will not always be the same sign as the LineSegment's Line
        direction since it depends on the start/end order and Line's does not.

        :getter: Returns the direction of the line segment.
        :setter: Read-only.
        """
        vector_ab = np.array(self.end) - np.array(self.start)
        unit_vector_ab = trig.get_unit_vector(vector_ab)
        return trig.to_1d_tuple(unit_vector_ab)

    @property
    def start(self) -> Point:
        """The start Point of the line segment.

        :getter: Returns the start point of the line segment.
        :setter: Updates the start point to match the location of a new point.
        :raises ValueError: When trying to update the point to a be at the same
            location as the end point or to a new dimension.
        """
        return self._start

    @start.setter
    def start(self, pt: Point) -> None:
        if self.end.is_equal(pt):
            msg = "Cannot set start to the same location as the LineSegment's end."
            raise ValueError(msg)
        self._start.update(pt)

    @property
    def end(self) -> Point:
        """The end Point of the line segment.

        :getter: Returns the end point of the line segment.
        :setter: Updates the end point to match the location of a new point.
        :raises ValueError: When trying to update the point to a be at the same
            location as the start point or to a new dimension.
        """
        return self._end

    @end.setter
    def end(self, pt: Point) -> None:
        if self.start.is_equal(pt):
            msg = "Cannot set end to the same location as the LineSegment's start."
            raise ValueError(msg)
        self._end.update(pt)

    # Public Methods #
    def copy(self) -> LineSegment:
        """Returns a copy of the LineSegment.

        :returns: A new LineSegment with new start and end points at the same
            position as this LineSegment, but with no uids assigned.
        """
        return LineSegment(self.start.copy(), self.end.copy())

    def is_equal(self, other: LineSegment) -> bool:
        return self.start.is_equal(other.start) and self.end.is_equal(other.end)

    def get_line(self) -> Line:
        """Returns the infinite Line coincident with the start and end of the
        LineSegment.
        """
        return Line.from_two_points(self.start, self.end)

    def update(self, other: LineSegment) -> Self:
        """Updates the points of the line segment to match the points of another
        line segment.

        :param other: The line segment to update to.
        :returns: The updated LineSegment.
        """
        self.start.update(other.start)
        self.end.update(other.end)
        return self

    # Python Dunders
    def __conform__(self, protocol: Type[PrepareProtocol]) -> str:
        if protocol is PrepareProtocol:
            return ";".join(map(str, [*self.start, *self.end]))
        raise TypeError(f"Expected sqlite3.PrepareProtocol, got {protocol}")

    def __copy__(self) -> LineSegment:
        """Returns a copy of the LineSegment that has the same points and line,
        but no assigned uid.
        """
        return self.copy()

    def __len__(self) -> int:
        """Returns the number of elements in the line segment's start, which
        is equivalent to the line segment's number of dimnesions.
        """
        return len(self.start)

    def __repr__(self) -> str:
        pt_a_strs, pt_b_strs = [], []
        for i in range(0, len(self)):
            if np.isclose(self.start[i], 0):
                pt_a_strs.append("0")
            else:
                pt_a_strs.append(f"{self.start[i]:g}")
            if np.isclose(self.end[i], 0):
                pt_b_strs.append("0")
            else:
                pt_b_strs.append(f"{self.end[i]:g}")
        start_str = ",".join(pt_a_strs)
        end_str = ",".join(pt_b_strs)
        return super().__repr__().format(
            details=f"({start_str})({end_str})"
        )
