import unittest

from conveyor_counter.geometry import Line2D, Point
from conveyor_counter.tracker import CentroidTracker, LineCrossingCounter


class TestTracker(unittest.TestCase):
    def test_tracker_assignments(self):
        tr = CentroidTracker(max_distance=50, max_missing=2)
        # frame 1
        tr.update([(Point(10, 10), (5, 5, 10, 10), "Red")])
        self.assertEqual(len(tr.tracks), 1)
        tid = next(iter(tr.tracks.keys()))

        # frame 2: close centroid => same id
        tr.update([(Point(15, 12), (10, 7, 10, 10), "Red")])
        self.assertIn(tid, tr.tracks)

    def test_color_mismatch_penalty(self):
        tr = CentroidTracker(max_distance=50, max_missing=2)
        tr.update([(Point(10, 10), (5, 5, 10, 10), "Red")])
        tid = next(iter(tr.tracks.keys()))

        # Same position but different color — should create a new track, not reuse ID
        tr.update([(Point(12, 12), (7, 7, 10, 10), "Blue")])
        self.assertEqual(len(tr.tracks), 2)
        self.assertIn(tid, tr.tracks)

    def test_line_cross_count_once(self):
        tracker = CentroidTracker(max_distance=100, max_missing=2)
        counter = LineCrossingCounter()
        line = Line2D(0, 0, 10, 0)

        tracker.update([(Point(5, 1), (0, 0, 10, 10), "Green")])
        prev = {tid: tr.centroid for tid, tr in tracker.tracks.items()}

        tracker.update([(Point(5, -1), (0, 0, 10, 10), "Green")])
        counter.update_counts(tracker.tracks, prev, line)
        self.assertEqual(counter.total, 1)

        # cross again should not increase (track counted)
        prev2 = {tid: tr.centroid for tid, tr in tracker.tracks.items()}
        tracker.update([(Point(5, 1), (0, 0, 10, 10), "Green")])
        counter.update_counts(tracker.tracks, prev2, line)
        self.assertEqual(counter.total, 1)


if __name__ == "__main__":
    unittest.main()
