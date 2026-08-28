"""A module providing a constraint classes for snapto constraints in 2D
geometry contexts. pancad defines a snapto constraint as one that can be applied
to geometry with no additional arguments but still meaningfully constrain the
geometry.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from pancad.abstract import AbstractConstraint
from pancad.constants import SketchConstraint

if TYPE_CHECKING:
    from uuid import UUID

    from pancad.abstract import AbstractGeometry, AbstractGeometrySystem
    from pancad.constants import ConstraintReference

class AbstractSingleSnapTo(AbstractConstraint):
    """An abstract class of constraints that can be applied to a single geometry element without
    any further definition.
    """
    def __init__(self, *geometry: AbstractGeometry,
                 uid: UUID | str | None=None, system: AbstractGeometrySystem | None=None,
                 name: str | None=None) -> None:
        self.uid = uid
        super().__init__(system, name=name)
        if len(geometry) != 1:
            raise ValueError(f"Expected 1 geometry, got: {geometry}")
        self._geometry = geometry

class Fixed(AbstractSingleSnapTo):
    """A class of constraint that can be applied to geometry to lock down its location,
    orientation, and any size parameters.
    """

    type_name = SketchConstraint.FIXED

class Unique(AbstractSingleSnapTo):
    """A class of constraint that can be applied to geometry to enforce unique vector constraints
    on element direction vectors. For example: An Axis direction would be held unique by setting
    its direction to the same as its reference Line. A Plane would be held unique by using the
    same rules as Line.
    """

    type_name = SketchConstraint.UNIQUE

class AbstractSnapTo(AbstractConstraint):
    """An abstract class of constraints that can be applied to a set of **one
    or two** geometries without any further definition.

    :param geometry: The a single or pair of geometry element(s) to constrain.
    :param uid: The unique id of the constraint.
    """
    def __init__(self, *geometry: AbstractGeometry,
                 uid: UUID| str | None=None, system: AbstractGeometrySystem | None=None,
                 name: str | None=None) -> None:
        self.uid = uid
        super().__init__(system, name=name)
        if len(geometry) not in [1, 2]:
            raise ValueError(f"Expected 1 or 2 geometries, got: {geometry}")
        if any(len(g) != 2 for g in geometry):
            non_two_dimensional = [g for g in geometry if len(g) != 2]
            raise ValueError(f"Non-2D Geometry provided: {non_two_dimensional}")
        self._geometry = geometry

class Horizontal(AbstractSnapTo):
    """A constraint that sets either a single geometry horizontal or a pair of
    geometries horizontal relative to each other in a 2D coordinate system. Can
    constrain:
    """

    type_name = SketchConstraint.HORIZONTAL

class Vertical(AbstractSnapTo):
    """A constraint that sets either a single geometry vertical or a pair of
    geometries vertical relative to each other in a 2D coordinate system. Can
    constrain:
    """

    type_name = SketchConstraint.VERTICAL
