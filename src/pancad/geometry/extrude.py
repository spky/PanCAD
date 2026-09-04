"""A module providing a class to represent linear extrusions in 3D space 
starting from a sketch profile.
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from pancad.abstract import AbstractFeature
from pancad.constants import FeatureType
from pancad.utils.initialize import get_pancad_config
from pancad.geometry.unique_lists import FeatureGeometryList

if TYPE_CHECKING:
    from numbers import Real
    from typing import NoReturn, Self
    from pancad.abstract import AbstractGeometry, AbstractFeatureSystem
    from pancad.geometry.sketch import Sketch

DEFAULT_NAME = get_pancad_config()["features"]["default_names"]["extrude"]

@dataclasses.dataclass
class ExtrudeSettings:
    """A dataclass containing the constant settings for an Extrude feature.

    :param type_: The active type of the Extrude that sets whether the extrude 
        is linear, midplane, etc.
    :param length: The length of the Extrude when a static length is active.
    :param opposite_length: The opposite length of the Extrude when a static 
        opposite length is active. Ignored if the Extrude is midplane.
    :param unit: The unit of the Extrude's length values.
    :param taper_angle: The angle that the default length's extrusion tapers in 
        degrees. A positive angle flares the extrude out while a negative angle 
        tapers the extrude inwards.
    :param opposite_taper_angle: The equivalent of the taper_angle, but for the 
        opposite side.
    :raises ValueError: When a negative number is provided for length or 
        opposite_length.
    """
    type_: FeatureType
    length: Real = 0
    opposite_length: Real = 0
    taper_angle: Real = 0
    opposite_taper_angle: Real = 0
    unit: str = None
    def __post_init__(self):
        if self.length < 0:
            raise ValueError(f"length cannot be <0, got {self.length}")
        if self.opposite_length < 0:
            raise ValueError("opposite_length cannot be <0,"
                             f" got {self.opposite_length}")
        if self.type_ in [FeatureType.UP_TO_LAST, FeatureType.UP_TO_FIRST]:
            raise NotImplementedError("Condition Extrudes not supported yet")
        if self.type_ in [FeatureType.UP_TO_FACE, FeatureType.UP_TO_BODY]:
            raise NotImplementedError("End Feature Extrudes not supported yet")

class Extrude(AbstractFeature):
    """A class representing linear extrusions starting from a sketch profile in 
    3D space. Extrusions in CAD programs often store suppressed information that 
    can be recalled later, such as when a two dimension extrusion is changed 
    to a one dimension extrusion and back again. Extrude also saves this 
    suppressed information and uses a FeatureType enumeration value to 
    determine which information is relevant currently.

    :param profile: The sketch defining the extrusion 2D shape.
    :param settings: The constant settings of the Extrude, stored inside a 
        ExtrudeSettings dataclass.
    :param name: The human-visible name of the extrusion.
    :param uid: The unique id of the Extrude. When set to None the uid is 
        automatically generated.
    """
    VALUE_STR_FORMAT = "{value}{unit}"
    # Feature Type Checking Class Constants #
    def __init__(self, profile: Sketch, settings: ExtrudeSettings,
                 *,
                 name: str=DEFAULT_NAME, uid: str=None,
                 system: AbstractFeature=None) -> None:
        super().__init__(system, name)
        self.uid = uid
        self.profile = profile
        self.settings = settings
        self._feature_geometry = FeatureGeometryList(self, [])

    # Class Methods #
    @classmethod
    def from_length(cls, profile: Sketch, length: Real, *,
                    name: str=DEFAULT_NAME, uid: str=None,
                    system: AbstractFeature=None, **settings) -> Self:
        """Initializes a linear extrude from length dimensions.

        :param profile: The Sketch feature defining the extrusion's 2D shape.
        :param length: The length of the extrusion the direction specified by 
            the extrusion type relative to the profile sketch.
        :param name: The human-visible name of the extrusion.
        :param uid: The unique id of the Extrude. Defaults to None.
        :param settings: See :class:`~pancad.geometry.extrude.ExtrudeSettings` 
            for additional keyword arguments. If the type setting is not 
            provided, it's set to `~pancad.api.FeatureType.DIMENSION`.
        """
        type_ = settings.setdefault("type_", FeatureType.DIMENSION)
        if type_ not in FeatureType.SINGLE_DIMENSION:
            raise TypeError(f"type_ '{type_}' needs more than one length.")
        settings.pop("type_", None)
        return cls(profile, ExtrudeSettings(type_, length, **settings),
                   name=name, uid=uid, system=system)

    # Properties #

    @property
    def feature_geometry(self) -> FeatureGeometryList:
        return self._feature_geometry

    @property
    def length(self) -> Real:
        """The linear length of the extrude in its normal direction."""
        return self.settings.length
    @length.setter
    def length(self, value: Real) -> None:
        self.settings = dataclasses.replace(self.settings, length=value)

    @property
    def opposite_length(self) -> Real:
        """The linear length of the extrude opposite its normal direction."""
        return self.settings.opposite_length
    @opposite_length.setter
    def opposite_length(self, value: Real) -> None:
        self.settings = dataclasses.replace(self.settings, opposite_length=value)

    @property
    def type_(self) -> FeatureType:
        """The active type of extrusion method."""
        return self.settings.type_
    @type_.setter
    def type_(self, value: FeatureType) -> None:
        self.settings = dataclasses.replace(self.settings, type_=value)

    @property
    def taper_angle(self) -> Real:
        """The angle the extrude tapers in its default direction in degrees."""
        return self.settings.taper_angle
    @taper_angle.setter
    def taper_angle(self, value: Real) -> None:
        self.settings = dataclasses.replace(self.settings, taper_angle=value)

    @property
    def opposite_taper_angle(self) -> Real:
        """The angle the extrude tapers in its opposite direction in degrees."""
        return self.settings.opposite_taper_angle
    @opposite_taper_angle.setter
    def opposite_taper_angle(self, value: Real) -> None:
        self.settings = dataclasses.replace(self.settings,
                                            opposite_taper_angle=value)

    @property
    def unit(self) -> str | None:
        """The unit of the Extrude's length values."""
        return self.settings.unit
    @unit.setter
    def unit(self, value: str) -> None:
        self.settings = dataclasses.replace(self.unit, unit=value)

    # Public Methods #
    def get_dependencies(self) -> list[AbstractFeature]:
        dependencies = set(super().get_dependencies())
        dependencies.add(self.profile)
        return list(dependencies)

    def get_length_string(self) -> str:
        """Return length value with the associated unit."""
        return self._get_value_string(self.length)

    def get_opposite_length_string(self) -> str:
        """Return opposite length value with the associated unit."""
        return self._get_value_string(self.opposite_length)

    def is_equal(self, other: Extrude) -> bool:
        return (self.profile.is_equal(other.profile)
                and self.settings == other.settings)

    # Private Methods #
    def _get_value_string(self, value: Real | None) -> str:
        """Returns a string of the constraint's value with the constraint's 
        unit. If the unit is None, then this just returns the value as a string.
        """
        if value is None:
            return ""
        if self.unit is None:
            return str(value)
        return self.VALUE_STR_FORMAT.format(value=value, unit=self.unit)

    # Python Dunders #
    def __repr__(self) -> str:
        return super().__repr__().format(details=f"'{self.name}'{self.type_!r}")
