"""A module providing a class that will read FreeCAD files and convert 
information in them into equvalent SVGs
"""

from __future__ import annotations

import Sketcher

import svg.elements as se
import freecad.sketch_readers as fsr
import translators.freecad_sketcher_to_svg as fc_to_svg
import translators.svg_to_freecad_sketcher as svg_to_fc
import svg.element_utils as seu

class SketchSVG(se.svg):
    """A class representing svg elements containing FreeCAD sketch information.
    """
    
    point_radius = "0.1"
    
    def __init__(self) -> None:
        """Constructor method"""
        self.geometry_g = None
        super().__init__()
    
    def add_geometry(self, fc_geometry: dict) -> None:
        """Adds geometry from a FreeCAD geometry dictionary to the svg's 
        geometry group.
        
        :param fc_geometry: A dictionary containing FreeCAD geometry info
        """
        geometry_type = fc_geometry["geometry_type"]
        match geometry_type:
            case "line" | "circular_arc":
                self.geometry_g.append(se.path(fc_geometry["id"],
                                               fc_geometry["d"])
                )
            case "point":
                self.geometry_g.append(se.circle(fc_geometry["id"],
                                                 fc_geometry["cx"],
                                                 fc_geometry["cy"],
                                                 self.point_radius)
                )
            case "circle":
                self.geometry_g.append(se.circle(fc_geometry["id"],
                                                 fc_geometry["cx"],
                                                 fc_geometry["cy"],
                                                 fc_geometry["r"])
                )
            case _:
                raise ValueError(f"'{geometry_type}' is not supported")
    
    @property
    def geometry(self) -> list[dict]:
        """Returns svg geometry from all the svg's subelements. Read-only.
        
        :returns: a list of svg geometry dictionaries
        """
        geo_list = []
        for shape in list(self.geometry_g):
            shape_list = []
            for geo_dict in shape.geometry:
                # points are represented as circles, so they need labeling
                if geo_dict["id"].startswith("point"):
                    geo_dict["geometry_type"] = "point"
                shape_list.append(geo_dict)
            geo_list.extend(shape_list)
        return geo_list
    
    def get_freecad_dict(self) -> list[dict]:
        """Returns FreeCAD geometry from all the svg's subelements.
        
        :returns: a list of FreeCAD geometry dictionaries
        """
        return svg_to_fc.translate_geometry(self.geometry)
    
    @classmethod
    def from_sketch(cls, sketch: Sketcher.Sketch,
                    unit: str) -> FreeCADSketchSVG:
        """Returns a new SketchSVG made from a FreeCAD Sketch
        
        :param sketch: A FreeCAD Sketcher.Sketch object to convert to svg
        :param unit: The length unit used by the sketch
        :returns: A new FreeCAD SketchSVG object
        """
        new_sketch_svg = cls()
        new_sketch_svg.unit = unit
        new_sketch_svg.Label = sketch.Label
        # Non-construction group
        new_sketch_svg.geometry_g = se.g(new_sketch_svg.Label
                                         + "_geometry")
        new_sketch_svg.append(new_sketch_svg.geometry_g)
        freecad_geometry = fsr.read_sketch_geometry(sketch)
        svg_geometry = fc_to_svg.translate_geometry(freecad_geometry)
        for geometry in svg_geometry:
            new_sketch_svg.add_geometry(geometry)
        new_sketch_svg.auto_size()
        return new_sketch_svg
    
    @classmethod
    def from_element(cls, svg_element: se.svg) -> FreeCADSketchSVG:
        """Returns a FreeCADSketchSVG made from an svg element
        
        :param svg_element: A svg element to make a FreeCAD sketch from
        :returns: A new FreeCAD SketchSVG object
        """
        new_sketch_svg = super().from_element(svg_element)
        for sub in list(svg_element):
            new_sketch_svg.append(seu.upgrade_element(sub))
        new_sketch_svg.Label = new_sketch_svg.id_
        new_sketch_svg.geometry_g = new_sketch_svg.sub(
            new_sketch_svg.Label + "_geometry"
        )
        if new_sketch_svg.geometry_g is None:
            raise ValueError("No geometry group found in svg")
        return new_sketch_svg