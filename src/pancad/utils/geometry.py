"""A module providing helper functions for defining geometry and constraints."""
from __future__ import annotations

from functools import wraps
from collections.abc import Sequence, Collection
from typing import TYPE_CHECKING, overload
from warnings import catch_warnings

import numpy as np

from pancad.utils import trigonometry as trig
from pancad.utils.quat import Quat

if TYPE_CHECKING:
    from collections.abc import Callable, Sized
    from typing import Any, ParamSpec, TypeVar, Concatenate

    import numpy.typing as npt

    from pancad.utils.pancad_types import SpaceVector, Space3DVector, Numpy1D

    P = ParamSpec("P")
    S = TypeVar("S", bound=Sized)
    A = TypeVar("A", bound=Sized)
    R = TypeVar("R")

### Wrappers
def dimension_bounded(bound: int) -> Callable[[Callable[Concatenate[S, P], R]],
                                              Callable[Concatenate[S, P], R]]:
    """A wrapper to raise an error when a method that only works in 2D/3D is called on 3D/2D
    geometry.

    :param bound: The dimension to check for. Ex: When provided 2, any method called with 3D
        geometry will raise a ValueError.
    """
    def bounder(func: Callable[Concatenate[S, P], R]) -> Callable[Concatenate[S, P], R]:
        @wraps(func)
        def wrapper(obj: S, /, *args: P.args, **kwargs: P.kwargs) -> R:
            if len(obj) != bound:
                raise ValueError(f"{func.__name__} Method only available on {bound}D"
                                 f" {obj.__class__.__name__}s")
            result = func(obj, *args, **kwargs)
            return result
        return wrapper
    return bounder


def no_dimensional_mismatch(func: Callable[Concatenate[S, A, P], R]
                            ) -> Callable[Concatenate[S, A, P], R]:
    """A wrapper to raise an error when the first argument of a method does not
    match the dimension of the geometry.
    """
    @wraps(func)
    def wrapper(obj: S, value: A, /, *args: P.args, **kwargs: P.kwargs) -> R:
        if len(obj) != len(value):
            raise ValueError(
                "Input Dimensional Mismatch:"
                f" {len(obj)}D {obj.__class__.__name__}"
                f" and {len(value)}D {value.__class__.__name__}"
            )
        result = func(obj, value, *args, **kwargs)
        return result
    return wrapper

### Functions
@overload
def get_unique_vector(vector: Sequence[float]) -> tuple[float, ...]: ...
@overload
def get_unique_vector(vector: Numpy1D) -> Numpy1D: ...
def get_unique_vector(vector: Sequence[float] | Numpy1D) -> tuple[float, ...] | Numpy1D:
    """Checks the vector against unique direction rules and inverts it if any are violated.

    Example of the algorithm using 3D vectors:
    1. The z component must be nonnegative.
    2. If z is exactly 0 or the vector is 2D, y must be nonnegative.
    3. If both y and z are exactly 0 or the vector is 2D, x must be nonnegative.
    4. Zero vectors are considered already unique and returned as is.

    :param vector: An n-dimensional vector.
    """
    tuple_vector = tuple(vector)
    for component in tuple_vector[::-1]:
        if component < 0:
            tuple_vector = tuple(map(lambda c: -c, tuple_vector))
            break
        if component > 0:
            break
    # Add 0 to ensure negative zero representations are eliminated
    tuple_vector = tuple(map(lambda c: c + 0, tuple_vector))
    if isinstance(vector, np.ndarray):
        return np.array(tuple_vector)
    return tuple_vector

def parse_vector(*components: float | Collection[float]) -> SpaceVector:
    """Batches structures of vector component inputs to a tuple of Reals.
    Usually used by pancad to parse position and direction information into the
    geometry classes.

    :raises TypeError: When provided a single component that is not a Collection or
        when 2 or more non-Real arguments.
    :raises ValueError: When provided 0 or more than 3 arguments.
    """
    tuple_vector: tuple[float, ...] | None = None
    if len(components) == 1: # Collection case
        vector = components[0]
        if isinstance(vector, np.ndarray):
            if vector.shape not in [(2,), (3,), (2, 1), (3, 1)]:
                raise ValueError(f"NumPy vectors must be 2 or 3 elements, got {vector}")
            tuple_vector = tuple(float(component.squeeze()) for component in vector)
        elif isinstance(vector, Collection):
            tuple_vector = tuple(vector)
        else:
            raise TypeError(f"Expected a Collection, got: {type(components)}")
    if len(components) in {2, 3}: # Starred args case
        floats = [c for c in components if isinstance(c, (int, float))]
        if len(floats) != len(components):
            types = [type(component) for component in components]
            raise TypeError(f"Expected only int/float components, got: {types}")
        tuple_vector = tuple(floats)
    if tuple_vector:
        assert len(tuple_vector) == 2 or len(tuple_vector) == 3
        return tuple_vector
    raise TypeError(f"Expected 1 to 3 components, got {components}")

def closest_to_origin(point: SpaceVector, vector: SpaceVector) -> Numpy1D:
    """Returns the point closest to the origin on a line created by a point
    and a vector.

    :param point: A vector to a point on the line.
    :param vector: A vector in the same direction as the line.
    :returns: A numpy array vector pointing to the closest point on the line.
    :raises ValueError: When the direction vector is a zero vector or the
        point and vector dimensions do not match.
    """
    if np.allclose(vector, [0] * len(vector)):
        raise ValueError(f"Got zero vector for line direction: {tuple(vector)}")
    if len(point) != len(vector):
        raise ValueError(f"Point {point} and vector {vector} dimensions are not equal")
    point_vector = np.array(point)
    unit_vector = trig.get_unit_vector(vector)
    dot = float(np.dot(point_vector, unit_vector))
    if dot == 0:
        # Point vector and direction are perpendicular, or the point vector
        # is zero vector. Either way the provided point is the closest.
        return point_vector
    if np.isclose(abs(dot), np.linalg.norm(point_vector)):
        # Point vector and direction vector are parallel or anti-parallel,
        # so the closest point must be the origin.
        return np.array([0] * len(point))
    # No special case, so the off-closest point vector can be subtracted out
    # to get the closest point.
    return point_vector - dot * unit_vector

def get_perpendicular(vector: Space3DVector) -> Numpy1D:
    """Returns a non-unique 3D unit vector perpendicular to the vector by
    finding its cross product to the most orthogonal basis vector.

    :raises ValueError: When provided a zero vector.
    :raises TypeError: When provided a non-3D vector.
    """
    x, y, z = map(abs, vector)
    if np.allclose(vector, (0, 0, 0)):
        raise ValueError(f"Expected non-zero vector, got {vector}")
    x_axis = (1, 0, 0)
    y_axis = (0, 1, 0)
    z_axis = (0, 0, 1)
    ortho_map = {
        # x < y | x < z | y < z
        (True, True, False): x_axis,
        (True, True, True): x_axis,
        (True, False, False): z_axis,
        (True, False, True): z_axis,
        (False, False, True): y_axis,
        (False, True, True): y_axis,
        (False, True, False): z_axis,
        (False, False, False): z_axis,
    }
    ortho = np.cross(vector, ortho_map[x < y, x < z, y < z])
    return trig.get_unit_vector(ortho)

def get_rotation_quat(start: Space3DVector, target: Space3DVector) -> Quat:
    """Returns a (non-unique) shortest-arc quaternion to rotate the start vector
    to the target vector.

    :raises ValueError: When provided a zero vector.
    :raises TypeError: When provided a non-3D vector.
    """
    with catch_warnings(action="error"):
        # NumPy only produces a warning when a 2D cross product is attempted.
        try:
            scalar = np.linalg.norm(start) * np.linalg.norm(target) + np.dot(start, target)
            axis = np.cross(start, target)
        except (ValueError, DeprecationWarning) as exc:
            non_3d_msgs = ["2-dimensional vectors", "incompatible dimensions", "not aligned"]
            if any(non_3d in str(exc) for non_3d in non_3d_msgs):
                raise TypeError(f"start/target must be 3D, got: {start}, {target}") from exc
            raise
    if any(np.allclose(vector, (0, 0, 0)) for vector in [start, target]):
        raise ValueError(f"start/target cannot be zero vector: {start}, {target}")
    quat = Quat(scalar, *axis)
    norm = np.linalg.norm(quat)
    if np.isclose(norm, 0):
        # If the norm of the quaternion is 0, the vectors are anti-parallel.
        # Anti-parallel vectors have an infinite number of shortest arc
        # quaternions, so an arbitrary perpendicular vector must be used as a
        # rotation axis.
        axis = get_perpendicular(start)
        quat = Quat(scalar, *axis)
        norm = np.linalg.norm(quat)
    return quat / norm
