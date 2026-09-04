"""A module providing types specific to pancad tests. These are never to be used in the main
program.
"""

from typing import TypedDict, NamedTuple, TypeVar, Union

from pancad.constants import SketchConstraint
from pancad.utils.quat import Quat
from pancad.utils.pancad_types import SpaceVector

T = TypeVar("T")
TestGroup = tuple[list[str], list[T]] # A pairing of the list of test ids and the list of inputs.
ChangeTest = tuple["GeometrySampleData", "GeometrySampleData"] # Pair of initial and change data

class GeometrySampleData(TypedDict):
    """A dictionary containing inputs for an element of sample geometry.

    :params vectors: A mapping of 2 or 3 element long float vectors to names. Ex: Locations and
        directions.
    :params scalars: A dict of names to floats to specify geometry. Ex: Lengths and angles
    """
    vectors: dict[str, SpaceVector]
    scalars: dict[str, float]
    quats: dict[str, Quat]

class FeatureSampleData(TypedDict):
    """A dictionary containing inputs for any feature element.

    :param geometry: Geometry specifications for a sketch's geometry system.
    :param constraint: Constraint specifications for a sketch's geometry system.
    """
    pose: Union["GeometrySpec", None]
    geometry: tuple["GeometrySpec", ...]
    constraints: tuple["ConstraintSpec", ...]

class GeometrySpec(NamedTuple):
    """A NamedTuple with enough information to create a pancad geometry element for a test."""
    name: str
    type_: str
    construction: bool
    params: GeometrySampleData

class ConstraintSpec(NamedTuple):
    """A NamedTuple with enough information to create a pancad constraint element for a test."""
    name: str
    type_: SketchConstraint
    refs: tuple[str, ...]
    params: GeometrySampleData | None
    unit: str | None = None
    quadrant: int | None = None
    is_radians: bool | None = None

class FeatureSpec(NamedTuple):
    """A NamedTuple with enough information to create a pancad feature element for a test."""
    name: str
    type_: str
    params: FeatureSampleData
