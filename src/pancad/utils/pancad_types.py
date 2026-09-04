"""A module providing common types for use throughout pancad."""
from __future__ import annotations

from collections import namedtuple
from collections.abc import Sequence
from typing import Union, overload, TypedDict

import numpy as np

VectorLike = Union[Sequence[float], np.ndarray]
Space2DVector = tuple[float, float]
Space3DVector = tuple[float, float, float]

Numpy1D = np.ndarray[tuple[int], np.dtype[np.float64]]
"""A type for a horizontal 1D numpy float64 vector. The length of the numpy vector cannot be
checked statically."""

Numpy2D = np.ndarray[tuple[int, int], np.dtype[np.float64]]
"""A type for a 2D numpy float64 array."""

SpaceVector = Space2DVector | Space3DVector

class ConstraintKwargs(TypedDict, total=False):
    """A dictionary of all keyword arguments that can be used to create a pancad constraint that
    are specific to constraints.

    :param value: The constraint's associated value. Can be a length or an angle and is required
        for value constraints.
    :param unit: The unit used for the constraint.
    :param is_radians: Whether the value of an angle constraint is provided in radians.
    :param quadrant: The quadrant an angle constraint should appear in.
    """
    value: float | None
    unit: str | None
    quadrant: int | None
    is_radians: bool | None

class SqliteTomlConfig(TypedDict):
    """A dictionary of settings for storing pancad sqlite storage."""
    conform_type: dict[str, str]

class FeaturesTomlConfig(TypedDict):
    """A dictionary of settings for pancad CAD features."""
    default_names: dict[str, str]

class PancadTomlConfig(TypedDict):
    """The top level dictionary for the pancad.toml configuration file."""
    filenames: dict[str, str]
    paths: dict[str, str]
    sqlite: SqliteTomlConfig
    features: FeaturesTomlConfig

class PancadJsonCache(TypedDict, total=False):
    """The top level dictionary for pancad cache files."""
    application_paths: dict[str, str]

class PolarVector(Sequence[float]):
    """Class for tracking polar vector components."""

    @overload
    def __init__(self, *values: float) -> None: ...
    @overload
    def __init__(self, *values: Sequence[float]): ...
    def __init__(self, *values: float | Sequence[float]) -> None:
        if len(values) == 1 and isinstance(values[0], Sequence):
            r, phi = values[0]
        elif len(values) == 2:
            if isinstance(values[0], Sequence):
                raise TypeError("Unexpected nested sequence")
            if isinstance(values[1], Sequence):
                raise TypeError("Unexpected nested sequence")
            r = values[0]
            phi = values[1]
        else:
            raise ValueError("Expected one 2 long sequence or 2 components")
        self._values: tuple[float, float] = (r, phi)

    @property
    def r(self) -> float:
        """The polar vector radial value."""
        return self._values[0]

    @property
    def phi(self) -> float:
        """The polar vector angle value in radians."""
        return self._values[1]

    @overload
    def __getitem__(self, index: int) -> float: ...
    @overload
    def __getitem__(self, index: slice[int | None, int | None, int | None]) -> list[float]: ...
    def __getitem__(self, index: int | slice[int | None, int | None, int | None]
                    ) -> float | list[float]:
        return self._values[index]

    def __len__(self) -> int:
        return len(self._values)

    def __hash__(self) -> int:
        return hash(self._values)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PolarVector):
            return hash(self) == hash(other)
        if isinstance(other, tuple):
            return hash(self) == hash(other)
        return NotImplemented

    def __array__(self, dtype: None=None, copy: None=None) -> Numpy1D:
        array = np.array(self._values)
        if copy is not None and not copy:
            raise ValueError("PolarVector cannot return the original")
        if dtype:
            return array.astype(dtype)
        return array

    def __repr__(self) -> str:
        return str(self._values)

class SphericalVector(Sequence[float]):
    """Class for tracking spherical vector components."""

    @overload
    def __init__(self, *values: float) -> None: ...
    @overload
    def __init__(self, *values: Sequence[float]): ...
    def __init__(self, *values: float | Sequence[float]) -> None:
        if len(values) == 1 and isinstance(values[0], Sequence):
            r, phi, theta = values[0]
        elif len(values) == 3:
            if isinstance(values[0], Sequence):
                raise TypeError("Unexpected nested sequence")
            if isinstance(values[1], Sequence):
                raise TypeError("Unexpected nested sequence")
            if isinstance(values[2], Sequence):
                raise TypeError("Unexpected nested sequence")
            r = values[0]
            phi = values[1]
            theta = values[2]
        else:
            raise ValueError("Expected one 2 long sequence or 2 components")
        self._values: tuple[float, float, float] = (r, phi, theta)

    @property
    def r(self) -> float:
        """The polar vector radial value."""
        return self._values[0]

    @property
    def phi(self) -> float:
        """The spherical vector azimuth angle value in radians."""
        return self._values[1]

    @property
    def theta(self) -> float:
        """The spherical vector elevation angle value in radians."""
        return self._values[2]

    @overload
    def __getitem__(self, index: int) -> float: ...
    @overload
    def __getitem__(self, index: slice[int | None, int | None, int | None]) -> list[float]: ...
    def __getitem__(self, index: int | slice[int | None, int | None, int | None]
                    ) -> float | list[float]:
        return self._values[index]

    def __len__(self) -> int:
        return len(self._values)

    def __hash__(self) -> int:
        return hash(self._values)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SphericalVector):
            return hash(self) == hash(other)
        if isinstance(other, tuple):
            return hash(self) == hash(other)
        return NotImplemented

    def __array__(self, dtype: None=None, copy: None=None) -> Numpy1D:
        array = np.array(self._values)
        if copy is not None and not copy:
            raise ValueError("SphericalVector cannot return the original")
        if dtype:
            return array.astype(dtype)
        return array

FitBox2D = namedtuple("FitBox2D", ["min_", "max_"])
