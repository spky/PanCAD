"""A module providing constraint classes for state constraints between strictly
two geometry elements. pancad defines state constraints to be constraints
between 2 elements that have a constant implied value or no value associated
with them. Perpendicular constraints that force lines to be angled 90 degrees to
each other and equal constraints that force lines to be the same length are
examples of state constraints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pancad.abstract import AbstractConstraint
from pancad.constants import SketchConstraint

if TYPE_CHECKING:
    from uuid import UUID

    from pancad.abstract import AbstractGeometry, AbstractGeometrySystem

class AbstractStateConstraint(AbstractConstraint):
    """An abstract class for constraints that force **exactly two** geometry
    elements to share a constant implied value or some state.

    :param geometry: The geometries to be constrained.
    :param uid: The unique id of the constraint.
    """
    def __init__(self, *geometry: AbstractGeometry,
                 uid: UUID | str | None=None, system: AbstractGeometrySystem | None=None,
                 name: str | None=None) -> None:
        self.uid = uid
        super().__init__(system, name=name)
        if len(geometry) != 2:
            raise ValueError(f"Expected 2 geometries, provided {geometry}")
        self._geometry = geometry

class Coincident(AbstractStateConstraint):
    """A constraint that forces two geometry elements to occupy the same
    location.
    """

    type_name = SketchConstraint.COINCIDENT

class Equal(AbstractStateConstraint):
    """A constraint that forces two geometry elements to have the same
    context-specific value, such as two line segments sharing the same length.
    """

    type_name = SketchConstraint.EQUAL

class Parallel(AbstractStateConstraint):
    """A constraint that forces two geometry elements to be side by side and
    have the same distance continuously between them.
    """

    type_name = SketchConstraint.PARALLEL

class Perpendicular(AbstractStateConstraint):
    """A constraint that forces two geometry elements to be angled 90 degrees
    relative to each other.
    """

    type_name = SketchConstraint.PERPENDICULAR

class Tangent(AbstractStateConstraint):
    """A constraint that forces a line to touch a curve at a point while not
    also crossing the curve at that point.
    """

    type_name = SketchConstraint.TANGENT

class AlignAxes(AbstractStateConstraint):
    """A constraint that forces two coordinate systems to share the same
    location and the same respective axis directions or two axes to share the
    same axis directons. When used on two axes their locations may be different.
    """

    type_name = SketchConstraint.ALIGN_AXES

class Codirectional(AbstractStateConstraint):
    """A constraint the forces two axes to point in the same direction.
    The axes' locations may be different.
    """

    type_name = SketchConstraint.CODIRECTIONAL

class Antiparallel(AbstractStateConstraint):
    """A constraint that forces two axes to point in antiparallel directions. The axes' locations 
    may be different. The opposite of Antiparallel is Codirectional, not Parallel.
    """

    type_name = SketchConstraint.ANTIPARALLEL
