import unittest

from conveyor_counter.geometry import Line2D, Point, crossed_line


class TestGeometry(unittest.TestCase):
    def test_crossed_line_simple(self):
        line = Line2D(0, 0, 10, 0)  # x-axis
        self.assertFalse(crossed_line(Point(1, 1), Point(2, 1), line))
        self.assertTrue(crossed_line(Point(1, 1), Point(2, -1), line))

    def test_touch_is_crossing(self):
        line = Line2D(0, 0, 10, 0)
        self.assertTrue(crossed_line(Point(1, 1), Point(2, 0), line))


if __name__ == "__main__":
    unittest.main()
