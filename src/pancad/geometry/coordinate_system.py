"""A module providing a class to represent coordinate systems in CAD programs,
graphics, and other geometry use cases.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import numpy as np

from pancad.abstract import AbstractGeometry
from pancad.constants import ConstraintReference as CR
from pancad.geometry.point import Point
from pancad.geometry.line import Axis
from pancad.geometry.plane import Plane
from pancad.utils.trigonometry import yaw_pitch_roll
from pancad.utils.geometry import get_rotation_quat
from pancad.utils.text_formatting import format_vector
from pancad.utils.quat import Quat

if TYPE_CHECKING:
    from collections.abc import Collection
    from typing import Self, Literal
    from uuid import UUID

    from pancad.utils.pancad_types import SpaceVector, Numpy2D

class _CoordinateSystemRefs(TypedDict, total=False):
    # A TypedDict used to store all CoordinateSystem cross references in a type safe way.
    origin: Point
    x: Axis
    y: Axis
    z: Axis
    xy: Plane
    xz: Plane
    yz: Plane

class CoordinateSystem(AbstractGeometry):
    """A class representing coordinate systems in 2D and 3D space.

    :param origin: A 2D or 3D center Point of the coordinate system.
    :param rotation: A rotation matrix or quaternion to rotate the canonical 2D/3D
        CoordinateSystem to an orientation. Leaves the canonical system unrotated when None.
    :param uid: The unique ID of the coordinate system.
    """
    _axis_names: list[Literal["x", "y", "z"]] = ["x", "y", "z"]
    _plane_names: list[Literal["yz", "xz", "xy"]] = ["yz", "xz", "xy"]

    def __init__(self, origin: Collection[float], rotation: Numpy2D | Quat | None=None,
                 *, uid: UUID | str | None=None, name: str | None=None) -> None:
        self.uid = uid
        origin = Point(origin)
        vectors = [[0] * i + [1] + [0] * (len(origin) - i - 1) for i in range(len(origin))]
        self._sys_refs: _CoordinateSystemRefs = {"origin": origin}
        for axis_key, vector in zip(self._axis_names, vectors):
            self._sys_refs[axis_key] = Axis(origin, vector)
        if len(vectors) == 3: # 3D Coordinate System, add the planes.
            for plane_key, vector in zip(self._plane_names, vectors, strict=True):
                self._sys_refs[plane_key] = Plane(origin, vector)
        # Convert the sys_refs into a plane references dictionary. TypedDict items() are of type
        # object so the type needs to be checked with isinstance.
        references = {CR(k): v for k, v in self._sys_refs.items()
                      if isinstance(v, AbstractGeometry)}
        super().__init__({CR.CORE: self, **references}, name=name)
        if rotation is not None:
            self.rotate(rotation)

    @classmethod
    def from_yaw_pitch_roll(cls, position: Collection[float],
                            yaw: float=0, pitch: float=0, roll: float=0,
                            *, uid: str | None=None) -> Self:
        """Initializes a CoordinateSystem from yaw, pitch, and roll angles in
        radians.
        """
        rotation = yaw_pitch_roll(yaw, pitch, roll)
        return cls(position, rotation, uid=uid)

    # Properties
    @property
    def axes(self) -> dict[CR, Axis]:
        """The Axis elements of the CoordinateSystem mapped to the ConstraintReferences."""
        return {CR(r): self._sys_refs[r] for r in self._axis_names if r in self._sys_refs}

    @property
    def origin(self) -> Point:
        """The CoordinateSystem's Origin Point."""
        return self._sys_refs["origin"]

    @origin.setter
    def origin(self, point: Collection[float]) -> None:
        self.move_to_point(point)

    @property
    def planes(self) -> dict[CR, Plane]:
        """The Plane elements of the CoordinateSystem mapped to the ConstraintReferences."""
        return {CR(r): self._sys_refs[r] for r in self._plane_names if r in self._sys_refs}

    @property
    def x_axis(self) -> Axis:
        """The CoordinateSystem's X-Axis."""
        return self._sys_refs["x"]

    @property
    def y_axis(self) -> Axis:
        """The CoordinateSystem's Y-Axis."""
        return self._sys_refs["y"]

    @property
    def z_axis(self) -> Axis:
        """The CoordinateSystem's Z-Axis."""
        return self._sys_refs["z"]

    @property
    def xy_plane(self) -> Plane:
        """The CoordinateSystem's XY-Plane."""
        return self._sys_refs["xy"]

    @property
    def xz_plane(self) -> Plane:
        """The CoordinateSystem's XZ-Plane."""
        return self._sys_refs["xz"]

    @property
    def yz_plane(self) -> Plane:
        """The CoordinateSystem's YZ-Plane."""
        return self._sys_refs["yz"]

    # Public Methods
    def copy(self) -> CoordinateSystem:
        """Returns a copy of the CoordinateSystem.

        :returns: A CoordinateSystem with the same origin, axes, and planes, but
            not the same uid.
        """
        return CoordinateSystem(self.origin).update(self)

    def is_equal(self, other: CoordinateSystem) -> bool:
        comparisons = []
        for ref, geometry in self.children.items():
            if ref == CR.CORE:
                continue
            comparisons.append(geometry.is_equal(other.get_reference(ref)))
        return all(comparisons)

    def get_quaternion(self) -> Quat:
        """Returns a quaternion that can be used to rotate other vectors from
        the canonical cartesian coordinate system (1, 0, 0), (0, 1, 0),
        (0, 0, 1) to this coordinate system.
        """
        if len(self) == 2:
            msg = "Cannot return a quaternion for 2D CoordinateSystems"
            raise ValueError(msg)
        canon_cs = CoordinateSystem((0, 0, 0))
        quats = {}
        for ref, cs_axis in self.axes.items():
            canon_axis = canon_cs.axes[ref]
            assert len(canon_axis.direction) == 3 and len(cs_axis.direction) == 3
            quats[ref] = get_rotation_quat(canon_axis.direction, cs_axis.direction)
        quats = {ref: q for ref, q in quats.items() if not np.allclose(q, Quat(1, 0, 0, 0))}
        if not quats:
            # All quaternions were identity quaternions, so just return one.
            return Quat(1, 0, 0, 0)
        ref = next(iter(quats)) # Get one of the remaining references
        if np.allclose([q for _, q in quats.items()], [quats[ref]] * len(quats)):
            # Since identity quats have been filtered out, if all the leftover
            # rotations are almost equal then the either all the rotations were
            # equal or the rotation was around one of the canon axes.
            return quats[ref]
        if len(quats) == 2:
            # Special cases past here
            ref_1, ref_2 = tuple(quats)
            if np.allclose(quats[ref_1], -quats[ref_2]):
                # The the remaining axes are flipped, but have rotations that are
                # in opposite directions. Either will work.
                return quats[ref_1]
            if np.dot(quats[ref_1].vector, quats[ref_2].vector) == 0:
                # Both remaining axes must have been flipped around if the
                # rotation axes are perpendicular.
                shared_axis = np.cross(canon_cs.axes[ref_1].direction,
                                       canon_cs.axes[ref_2].direction)
                return Quat(0, *shared_axis)
        msg = ("Failed to find quaternion to rotate to this CoordinateSystem's"
               f" axes: {self.x_axis}, {self.y_axis}, {self.z_axis}."
               f" Leftover Quaternion Candidates: {quats}")
        raise NotImplementedError(msg)

    def update(self, other: CoordinateSystem) -> Self:
        """Updates the origin, axes, and planes of the CoordinateSystem to match
        another CoordinateSystem.

        :param other: The CoordinateSystem to update to.
        :returns: The updated CoordinateSystem.
        """
        for ref, geometry in self.children.items():
            if ref == CR.CORE:
                continue
            geometry.update(other.get_reference(ref))
        return self

    def rotate(self, rotation: Numpy2D | Quat) -> Self:
        """Rotates the system with a rotation matrix or quaternion."""
        for geometry in (self.axes | self.planes).values():
            geometry.rotate(rotation) # Rotate around closest points
            geometry.move_to_point(self.origin) # Realign axes and planes
        return self

    def move_to_point(self, location: Collection[float]) -> Self:
        """Moves the system to a new location with no rotation."""
        location = Point(location)
        self.origin.update(location)
        for geometry in (self.axes | self.planes).values():
            geometry.move_to_point(location)
        return self

    # Python Dunders
    def __copy__(self) -> CoordinateSystem:
        """Returns a copy of the CoordinateSystem that has the same origin,
        axes, planes and context, but not the same uid. Can be used with the
        python copy module.
        """
        return self.copy()

    def __repr__(self) -> str:
        label_map = {"": self.origin.cartesian,
                     "X": self.x_axis.direction, "Y": self.y_axis.direction}
        if len(self) == 3:
            label_map["Z"] = self.z_axis.direction
        strings = [f"{l}({format_vector(v)})" for l, v in label_map.items()]
        return super().__repr__().format(details="".join(strings))

    def __len__(self) -> int:
        """Returns the number of dimensions of the coordinate system by
        returning the number of dimensions of the origin point.
        """
        return len(self.origin)


class Pose(AbstractGeometry):
    """The position and orientation of a 3D object."""

    def __init__(self,
                 coordinate_system: CoordinateSystem,
                 *,
                 uid: str | UUID | None=None,
                 name: str | None=None) -> None:
        self.uid = uid
        if (dimensions := len(coordinate_system)) != 3:
            raise ValueError("Expected 3D coordinate system,"
                             f" got {dimensions}D: {coordinate_system}")
        self._coordinate_system = coordinate_system
        super().__init__(
            {
                CR.CORE: self,
                CR.ORIGIN: self._coordinate_system.origin,
                CR.X: self._coordinate_system.x_axis,
                CR.Y: self._coordinate_system.y_axis,
                CR.Z: self._coordinate_system.z_axis,
                CR.FRONT: self._coordinate_system.xy_plane,
                CR.RIGHT: self._coordinate_system.xz_plane,
                CR.TOP: self._coordinate_system.yz_plane,
                CR.CS: self._coordinate_system,
            },
            name=name
        )

    @classmethod
    def from_yaw_pitch_roll(cls, position: Collection[float],
                            yaw: float=0, pitch: float=0, roll: float=0,
                            uid: str | None=None) -> Self:
        """Initializes a Pose from yaw, pitch, and roll angles in radians."""
        coordinate_system = CoordinateSystem.from_yaw_pitch_roll(position, yaw, pitch, roll)
        return cls(coordinate_system, uid=uid)

    @classmethod
    def from_rotation(cls, origin: Collection[float], rotation: Numpy2D | Quat | None=None,
                      *, uid: UUID | str | None=None) -> Self:
        """Initializes a Pose using the origin point and a rotation around that point.

        :param origin: A 3D center Point of the Pose's coordinate system.
        :param rotation: A rotation matrix or quaternion to rotate a canonical 3D
            CoordinateSystem to an orientation. Leaves the canonical system unrotated when None.
        :param uid: The unique ID of the Pose.
        """
        return cls(CoordinateSystem(origin, rotation), uid=uid)

    @property
    def coordinate_system(self) -> CoordinateSystem:
        """Internal coordinate_system representing the the Pose."""
        return self._coordinate_system

    @property
    def origin(self) -> Point:
        """The origin point of the Pose's internal coordinate_system."""
        return self._coordinate_system.origin

    @property
    def front(self) -> Plane:
        """Front plane of the Pose."""
        return self._coordinate_system.xy_plane

    @property
    def right(self) -> Plane:
        """Right plane of the Pose."""
        return self._coordinate_system.xz_plane

    @property
    def top(self) -> Plane:
        """Top plane of the Pose."""
        return self._coordinate_system.yz_plane

    def is_equal(self, other: Pose) -> bool:
        return self.coordinate_system.is_equal(other.coordinate_system)

    def move_to_point(self, location: Collection[float]) -> Self:
        """Moves the Pose to a new location with no rotation."""
        self.coordinate_system.move_to_point(location)
        return self

    def rotate(self, rotation: Numpy2D | Quat) -> Self:
        """Rotates the pose with a rotation matrix or quaternion."""
        self.coordinate_system.rotate(rotation)
        return self

    def update(self, other: Pose) -> Self:
        """Updates the position and orientation of the Pose to the other Pose."""
        self.coordinate_system.update(other.coordinate_system)
        return self

    def __len__(self) -> int:
        """Returns the number of dimensions of the Pose. Poses are always 3D."""
        return 3

    def __repr__(self) -> str:
        origin = str(tuple(self.origin)).replace(" ", "")
        return super().__repr__().format(details=f"{origin}")
