from PanCAD.utils import initialize
from PanCAD.svg.file import read_svg
from PanCAD.freecad.object_wrappers import read_freecad
from PanCAD.translators.freecad_svg_file import freecad_sketch_to_svg
from PanCAD.translators.freecad_svg_file import svg_to_freecad_sketch

initialize.settings()