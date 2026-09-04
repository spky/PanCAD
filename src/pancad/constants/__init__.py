"""This module contains constants used throughout pancad."""

__all__ = [
    "AngleConvention",
    "ConfigCategory",
    "SoftwareName",
    "ConstraintReference",
    "FeatureType",
    "SketchConstraint",
    "ConstraintVariableName",
    "ConstraintEquationName",
]


from ._angle_convention import AngleConvention
from ._config_cache_category import ConfigCategory
from ._software_name import SoftwareName
from .constraint_reference import ConstraintReference
from .feature_type import FeatureType
from .sketch_constraint import SketchConstraint
from ._solver_constants import ConstraintVariableName, ConstraintEquationName

QUAL_DELIM = "::" # The delimiter for qualified names.
