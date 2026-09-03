"""A module providing functions to create PancadThing elements from test data file data."""
from __future__ import annotations

from typing import TYPE_CHECKING
import math

from pancad.constants import SketchConstraint
from pancad.geometry.coordinate_system import Pose
from pancad.geometry.point import Point
from pancad.geometry.line import Line
from pancad.geometry.sketch import Sketch

if TYPE_CHECKING:
    from collections.abc import Callable

    from pancad.abstract import AbstractFeature, AbstractGeometry
    from pancad.utils.pancad_types import ConstraintKwargs
    from tests._typing import GeometrySpec, FeatureSpec

def make_feature(spec: FeatureSpec) -> AbstractFeature:
    """Returns a feature per a FeatureSpec."""
    return _FEATURE_FUNCS[spec.type_](spec)

def make_geometry(spec: GeometrySpec) -> AbstractGeometry:
    """Returns a geometry element per a GeometrySpec."""
    return _GEOMETRY_FUNCS[spec.type_](spec)

# Feature Factories ##############################################################################
def _sketch(spec: FeatureSpec) -> Sketch:
    assert spec.params["pose"] is not None # pose is required for sketches.
    pose = _pose(spec.params["pose"])
    feature = Sketch(pose=pose, name=spec.name)
    feature.geometry_system.name = f"{spec.name}_system"
    for geometry in spec.params["geometry"]:
        feature.geometry_system.add_geometry(make_geometry(geometry), geometry.construction)
    for constraint in spec.params["constraints"]:
        kwargs: ConstraintKwargs = {"unit": constraint.unit, "quadrant": constraint.quadrant,
                                    "is_radians": constraint.is_radians}
        if constraint.params:
            value = constraint.params["scalars"].get("value")
            assert value is not None
            kwargs["value"] = value
            if constraint.type_ == SketchConstraint.ANGLE and not constraint.is_radians:
                kwargs["value"] = math.degrees(value)
        try:
            feature.geometry_system.constrain(constraint.type_, *constraint.refs,
                                              name=constraint.name, **kwargs)
        except (ValueError, LookupError) as exc:
            raise ValueError(f"Constraint '{constraint.name}' failed: {exc}") from exc
    return feature

# Geometry Factories #############################################################################
def _line(spec: GeometrySpec) -> Line:
    return Line(Point(spec.params["vectors"]["point"]), spec.params["vectors"]["direction"],
                name=spec.name)

def _pose(spec: GeometrySpec) -> Pose:
    return Pose.from_rotation(spec.params["vectors"]["origin"], spec.params["quats"]["rotation"],
                              name=spec.name)

def _point(spec: GeometrySpec) -> Point:
    return Point(spec.params["vectors"]["vector"], name=spec.name)

_FEATURE_FUNCS: dict[str, Callable[[FeatureSpec], AbstractFeature]] = {"sketch": _sketch}
_GEOMETRY_FUNCS: dict[str, Callable[[GeometrySpec], AbstractGeometry]] = {
    "point": _point, "line": _line,
}
