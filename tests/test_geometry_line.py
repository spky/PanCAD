import sys
from pathlib import Path
import unittest
import math
import copy

import numpy as np

sys.path.append('src')
from PanCAD.utils import trigonometry as trig
from PanCAD.geometry.point import Point
from PanCAD.geometry.line import Line
from PanCAD.utils import verification

ROUNDING_PLACES = 10

class TestLineInit(unittest.TestCase):
    
    def setUp(self):
        self.pt_a = Point((1,0,0))
        self.pt_b = Point((1,10,0))
    
    def test_line_init_no_arg(self):
        l = Line()
    
    def test_point_len_dunder(self):
        tests = [
            ((0, 0), (1, 1), 2),
            ((0, 0, 0), (1, 1, 1), 3),
        ]
        for pt_a, pt_b, length in tests:
            with self.subTest(point_a=pt_a, point_b=pt_b,
                              expected_length=length):
                point_1, point_2 = Point(pt_a), Point(pt_b)
                test_line = Line.from_two_points(point_1, point_2)
                self.assertEqual(len(test_line), length)
    
    def test_line_str_dunder(self):
        test = Line.from_two_points(self.pt_a, self.pt_b)
        expected = ("PanCAD Line with a point closest to the origin at"
                    + " (1.0, 0.0, 0.0) and in the direction (0.0, 1.0, 0.0)")
        self.assertEqual(str(test), expected)

class TestLineVectorMethods(unittest.TestCase):
    
    def test_unique_direction(self):
        tests = [
            ## 2D Tests ##
            ((0, 0), (0, 0)),
            # Positive Unit Vector
            ((1, 0), (1, 0)),
            ((0, 1), (0, 1)),
            # Negative Unit Vector
            ((-1, 0), (1, 0)),
            ((0, -1), (0, 1)),
            # 2 Direction Positive and Negative
            ((1, 1), trig.get_unit_vector(np.array((1, 1)))),
            ((-1, -1), trig.get_unit_vector(np.array((1, 1)))),
            ## 3D Tests ##
            # Zero Vector
            ((0, 0, 0), (0, 0, 0)),
            # Positive Unit Vector
            ((1, 0, 0), (1, 0, 0)),
            ((0, 1, 0), (0, 1, 0)),
            ((0, 0, 1), (0, 0, 1)),
            # Negative Unit Vector
            ((-1, 0, 0), (1, 0, 0)),
            ((0, -1, 0), (0, 1, 0)),
            ((0, 0, -1), (0, 0, 1)),
            # 2 Direction Positive
            ((1, 1, 0), trig.get_unit_vector(np.array((1,1,0)))),
            ((0, 1, 1), trig.get_unit_vector(np.array((0,1,1)))),
            # 2 Direction Negative
            ((-1, -1, 0), trig.get_unit_vector(np.array((1,1,0)))),
            ((0, -1, -1), trig.get_unit_vector(np.array((0, 1, 1)))),
            # 3 Direction Positive and Negative
            ((1, 1, 1), trig.get_unit_vector(np.array((1,1,1)))),
            ((-1, -1, -1), trig.get_unit_vector(np.array([1,1,1]))),
        ]
        tests_np = []
        for test in tests:
            test_np = []
            for element in test:
                test_np.append(np.array(element))
            tests_np.append(test_np)
        
        for vector, unit_vector in tests_np:
            with self.subTest(vector=vector, unit_vector=unit_vector):
                self.assertCountEqual(Line.unique_direction(vector),
                                      unit_vector)

class TestLineTwoPointDefinition(unittest.TestCase):
    
    def setUp(self):
        # Point A, Point B, Expected Point Closest to Origin,
        # Vector in Expected Direction (subsequently converted to unit vector)
        self.tests = [
            # Set 1: Diagonal Off-Origin
            ((0, 4), (4, 0), (2, 2), (-1, 1)),
            ((0, 4), (-4, 0), (-2, 2), (1, 1)),
            ((0, -4), (-4, 0), (-2, -2), (-1, 1)),
            ((0, -4), (4, 0), (2, -2), (1, 1)),
            # Set 2: Vertical +, -, and 0
            ((2, 0), (2, 2), (2, 0), (0, 1)),
            ((-2, 0), (-2, 2), (-2, 0), (0, 1)),
            ((0, 0), (0, 2), (0, 0), (0, 1)),
            # Set 3: Horizontal +, -, and 0
            ((0, 2), (2, 2), (0, 2), (1, 0)),
            ((0, -2), (2, -2), (0, -2), (1, 0)),
            ((0, 0), (2, 0), (0, 0), (1, 0)),
            # Set 4: Diagonal On-Origin Per Quadrant
            ((1, 1), (2, 2), (0, 0), (1, 1)),
            ((-1, 1), (-2, 2), (0, 0), (-1, 1)),
            ((-1, -1), (-2, -2), (0, 0), (1, 1)),
            ((1, -1), (2, -2), (0, 0), (-1, 1)),
            # Set 5: Diagonal On-Origin Across Quadrant
            ((-1, -1), (1, 1), (0, 0), (1, 1)),
            ((-1, 1), (1, -1), (0, 0), (-1, 1)),
            # Set 6: Horizontal/Vertical On-Origin Across Quadrant
            ((-1, 0), (1, 0), (0, 0), (1, 0)),
            ((0, -1), (0, 1), (0, 0), (0, 1)),
        ]
        for i, (pt_a, pt_b, e_pt, vector) in enumerate(self.tests):
            # Convert expected direction to unit vector
            np_vector = trig.to_1D_np(vector)
            unit_vector = trig.get_unit_vector(np_vector)
            self.tests[i] = (pt_a, pt_b, e_pt, trig.to_1D_tuple(unit_vector))
    
    def test_from_two_points_point_closest_to_origin(self):
        for point_a, point_b, expected_point, _ in self.tests:
            with self.subTest(point_a=point_a, point_b=point_b,
                              expected_point=expected_point):
                pt_a, pt_b = Point(point_a), Point(point_b)
                e_pt = Point(expected_point)
                test_line = Line.from_two_points(pt_a, pt_b)
                verification.assertPointsAlmostEqual(
                    self, test_line._point_closest_to_origin, e_pt,
                    ROUNDING_PLACES
                )
    
    def test_from_two_points_direction(self):
        for point_a, point_b, _, expected_direction in self.tests:
            with self.subTest(point_a=point_a, point_b=point_b,
                              expected_direction=expected_direction):
                pt_a, pt_b = Point(point_a), Point(point_b)
                test_line = Line.from_two_points(pt_a, pt_b)
                verification.assertTupleAlmostEqual(
                    self, test_line.direction, expected_direction,
                    ROUNDING_PLACES
                )
    
    def test_from_two_points_same_point(self):
        pt_a = Point((1, 1))
        pt_b = Point((1, 1))
        with self.assertRaises(ValueError):
            test_line = Line.from_two_points(pt_a, pt_b)

class TestEquationLineDefinitions(unittest.TestCase):
    
    def setUp(self):
        # Slope (m), Y-Intercept (b), Expected Point, Expected Direction
        self.tests = [
            (0, 0, (0, 0), (1, 0)),
            (1, 0, (0, 0), (1, 1)),
            (-1, 0, (0, 0), (-1, 1)),
            (-1, 4, (2, 2), (-1, 1)),
        ]
        for i, (m, b, pt, direction) in enumerate(self.tests):
            self.tests[i] = (m, b, Point(pt), trig.get_unit_vector(direction))
    
    def test_from_slope_and_y_intercept_expected_point(self):
        for m, b, pt, direction in self.tests:
            with self.subTest(slope=m, intercept=b,
                              expected_closest_to_origin_point=pt):
                test_line = Line.from_slope_and_y_intercept(m, b)
                verification.assertPointsAlmostEqual(
                    self, test_line._point_closest_to_origin, pt,
                    ROUNDING_PLACES
                )
    
    def test_from_slope_and_y_intercept_expected_direction(self):
        for m, b, pt, direction in self.tests:
            with self.subTest(slope=m, intercept=b,
                              expected_direction=direction):
                test_line = Line.from_slope_and_y_intercept(m, b)
                verification.assertTupleAlmostEqual(
                    self, test_line.direction, direction, ROUNDING_PLACES
                )
    
    def test_slope_getter_non_nan(self):
        for m, b, pt, direction in self.tests:
            with self.subTest(slope=m, intercept=b):
                test_line = Line.from_slope_and_y_intercept(m, b)
                self.assertEqual(test_line.slope, m)
    
    def test_y_intercept_getter_non_nan(self):
        for m, b, pt, direction in self.tests:
            with self.subTest(slope=m, intercept=b):
                test_line = Line.from_slope_and_y_intercept(m, b)
                self.assertEqual(test_line.y_intercept, b)

class TestLineCoordinateSystemConversion(unittest.TestCase):
    
    def setUp(self):
        """
        Test Order:
            Point A, Point B, Phi (Azimuth) Angle, Theta (Inclination) Angle
        Angles get converted to radians prior to test
        r separately defined for legibility since for line direction unit 
        vectors it will always be 1. 
        """
        tests = [
            ((0, 0, 0), (1, 0, 0), (1, 0, 90)),
            ((0, 0, 0), (0, 1, 0), (1, 90, 90)),
            ((0, 0, 0), (0, 0, 1), (1, math.nan, 0)),
        ]
        self.tests_2d, self.tests_3d = [], []
        for pt_a, pt_b, (r, phi, theta) in tests:
            self.tests_3d.append(
                (
                    Point(pt_a), Point(pt_b),
                    (r, math.radians(phi), math.radians(theta)),
                )
            )
            if pt_a[:2] != pt_b[:2]: # To deal with when x = y = 0 and z != 0
                self.tests_2d.append(
                    (Point(pt_a[:2]), Point(pt_b[:2]), (r, math.radians(phi)))
                )
    
    def test_direction_polar(self):
        for pt_a, pt_b, (r, phi) in self.tests_2d:
            with self.subTest(
                        point_a=tuple(pt_a), point_b=tuple(pt_b),
                        expected_phi=(f"Radians: {phi},"
                                     + f" Degrees: {math.degrees(phi)}")
                    ):
                test_line = Line.from_two_points(pt_a, pt_b)
                verification.assertTupleAlmostEqual(
                    self, test_line.direction_polar, (r, phi),
                    ROUNDING_PLACES
                )
    
    def test_direction_spherical(self):
        for pt_a, pt_b, (r, phi, theta) in self.tests_3d:
            with self.subTest(
                    point_a=tuple(pt_a), point_b=tuple(pt_b),
                    expected_phi=(f"Radians: {phi},"
                                 + f" Degrees: {math.degrees(phi)}"),
                    expected_theta=(f"Radians: {theta},"
                                    + f" Degrees: {math.degrees(theta)}")
                    ):
                test_line = Line.from_two_points(pt_a, pt_b)
                verification.assertTupleAlmostEqual(
                    self, test_line.direction_spherical, (r, phi, theta),
                    ROUNDING_PLACES
                )
    
    def test_direction_polar_setter(self):
        for pt_a, pt_b, polar_vector in self.tests_2d:
            with self.subTest(
                        point_a=tuple(pt_a), point_b=tuple(pt_b),
                        polar_vector=polar_vector
                    ):
                test_line = Line.from_two_points(pt_a, pt_b)
                before_direction = test_line.direction
                test_line.direction_polar = polar_vector
                verification.assertTupleAlmostEqual(
                    self, before_direction, test_line.direction,
                    ROUNDING_PLACES
                )
    
    def test_direction_spherical_setter(self):
        for pt_a, pt_b, spherical_vector in self.tests_3d:
            with self.subTest(
                        point_a=tuple(pt_a), point_b=tuple(pt_b),
                        spherical_vector=spherical_vector
                    ):
                test_line = Line.from_two_points(pt_a, pt_b)
                before_direction = test_line.direction
                test_line.direction_spherical = spherical_vector
                verification.assertTupleAlmostEqual(
                    self, before_direction, test_line.direction,
                    ROUNDING_PLACES
                )

class TestLineRichComparison(unittest.TestCase):
    
    def setUp(self):
        self.tests = [
            ((0, 0), (1, 1), (0, 0), (1, 1), True),
            ((1, 0), (1, 1), (0, 0), (1, 1), False),
            ((0, 0, 0), (1, 1, 1), (0, 0, 0), (1, 1, 1), True),
            ((1, 0, 0), (1, 1, 0), (0, 0, 0), (1, 1, 0), False),
        ]
        for i, (pt1a, pt1b, pt2a, pt2b, equality) in enumerate(self.tests):
            self.tests[i] = (Point(pt1a), Point(pt1b),
                             Point(pt2a), Point(pt2b),
                             equality)
    
    def test_line_equality(self):
        for pt1a, pt1b, pt2a, pt2b, expected_equality in self.tests:
            with self.subTest(point1a=tuple(pt1a), point1b=tuple(pt1b),
                              point2a=tuple(pt2a), point2b=tuple(pt2b),
                              expected_equality=expected_equality):
                line1 = Line.from_two_points(pt1a, pt1b)
                line2 = Line.from_two_points(pt2a, pt2b)
                self.assertEqual(line1 == line2, expected_equality)
    
    def test_line_inequality(self):
        for pt1a, pt1b, pt2a, pt2b, expected_equality in self.tests:
            with self.subTest(point1a=tuple(pt1a), point1b=tuple(pt1b),
                              point2a=tuple(pt2a), point2b=tuple(pt2b),
                              expected_equality= not expected_equality):
                line1 = Line.from_two_points(pt1a, pt1b)
                line2 = Line.from_two_points(pt2a, pt2b)
                self.assertEqual(line1 != line2, not expected_equality)

class TestLineCoincidence(unittest.TestCase):
    
    def setUp(self):
        coincidence_tests = [
            ((0, 0), (1, 1), (2, 2), True),
            ((0, 0), (1, 1), (0, 0), True),
            ((0, 0), (1, 1), (-2, -2), True),
            ((0, 0), (1, 1), (0, 1), False),
            ((-4, 0), (0, 4), (-2, 2), True),
            ((-4, 0), (0, 4), (-4, 0), True),
            ((-4, 0), (0, 4), (0, 4), True),
            ((-4, 0), (0, 4), (0, 0), False),
            ((0, -4), (4, 0), (0, 0), False),
            ((0, -4), (4, 0), (2, -2), True),
        ]
        
        self.tests_2d = []
        for pt_a, pt_b, check_pt, result in coincidence_tests:
            self.tests_2d.append(
                (Point(pt_a), Point(pt_b), Point(check_pt), result),
            )
    
    def test_is_coincident_2d(self):
        for pt_a, pt_b, check_pt, result in self.tests_2d:
            with self.subTest(point_a=tuple(pt_a), point_b=tuple(pt_b),
                              check_pt=tuple(check_pt), expected_result=result):
                test_line = Line.from_two_points(pt_a, pt_b)
                self.assertEqual(test_line.is_coincident(check_pt), result)

class TestLineParallel(unittest.TestCase):
    
    def setUp(self):
        tests = [
            ((0, 0), (1, 1), (0, 1), (1, 2), True),
            ((0, 1), (0, 2), (1, 1), (1, 2), True),
            ((0, 0), (1, 0), (0, 1), (1, 1), True),
            ((0, 0), (1, 0), (1, 0), (1, 2), False),
        ]
        
        self.tests = []
        for pt_1a, pt_1b, pt_2a, pt_2b, parallel in tests:
            self.tests.append(
                (
                    Line.from_two_points(Point(pt_1a), Point(pt_1b)),
                    Line.from_two_points(Point(pt_2a), Point(pt_2b)),
                    parallel
                )
            )
    
    def test_is_parallel(self):
        for line1, line2, parallel in self.tests:
            with self.subTest(line1=str(line1), line2=str(line2),
                              expected_result=parallel):
                self.assertEqual(line1.is_parallel(line2), parallel)

class TestLinePointMovers(unittest.TestCase):
    
    def setUp(self):
        tests = [
            ((0, 1), (1, 1), (0, 2)),
            ((0, 1), (1, 1), (3, 3)),
            ((-4, 0), (0, 4), (3, 3)),
        ]
        self.tests_2d = []
        for pt_a, pt_b, new_pt in tests:
            self.tests_2d.append(
                (Point(pt_a), Point(pt_b), Point(new_pt))
            )
    
    def test_move_to_point_2d(self):
        for pt_a, pt_b, new_pt in self.tests_2d:
            with self.subTest(point_a=tuple(pt_a), point_b=tuple(pt_b),
                              new_point=tuple(new_pt)):
                original_line = Line.from_two_points(pt_a, pt_b)
                test_line = copy.copy(original_line)
                test_line.move_to_point(new_pt)
                results = [
                    test_line.is_coincident(new_pt),
                    test_line.is_parallel(original_line)
                ]
                self.assertTrue(all(results))

if __name__ == "__main__":
    with open("tests/logs/" + Path(sys.modules[__name__].__file__).stem
              +".log", "w") as f:
        f.write("finished")
    unittest.main()