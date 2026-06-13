from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Line2D, Point, crossed_line


@dataclass
class Track:
    track_id: int
    centroid: Point
    bbox: tuple[int, int, int, int]  # x,y,w,h
    missing: int = 0
    counted: bool = False
    color: str = "unknown"


class CentroidTracker:
    """Simple nearest-centroid tracker.

    Works well for conveyor belts where objects move mostly in one direction
    and do not cross frequently.
    """

    def __init__(self, max_distance: float = 60.0, max_missing: int = 15):
        self.max_distance = float(max_distance)
        self.max_missing = int(max_missing)
        self._next_id = 1
        self.tracks: dict[int, Track] = {}

    def reset(self) -> None:
        self._next_id = 1
        self.tracks.clear()

    def _new_track(self, centroid: Point, bbox: tuple[int, int, int, int], color: str = "unknown") -> Track:
        t = Track(track_id=self._next_id, centroid=centroid, bbox=bbox, color=color)
        self._next_id += 1
        self.tracks[t.track_id] = t
        return t

    def update(self, detections: list[tuple[Point, tuple[int, int, int, int], str] | tuple[Point, tuple[int, int, int, int]]]) -> dict[int, Track]:
        """Update tracks with detections (centroid, bbox, color)."""
        normalized_dets = []
        for d in detections:
            if len(d) == 2:
                normalized_dets.append((d[0], d[1], "unknown"))
            else:
                normalized_dets.append((d[0], d[1], d[2]))

        # No existing tracks: create all
        if not self.tracks:
            for c, b, col in normalized_dets:
                self._new_track(c, b, col)
            return self.tracks

        track_ids = list(self.tracks.keys())
        track_centroids = np.array([[self.tracks[i].centroid.x, self.tracks[i].centroid.y] for i in track_ids], dtype=np.float32)

        det_centroids = np.array([[d[0].x, d[0].y] for d in normalized_dets], dtype=np.float32)

        # If no detections, mark missing
        if len(normalized_dets) == 0:
            to_del = []
            for tid in track_ids:
                self.tracks[tid].missing += 1
                if self.tracks[tid].missing > self.max_missing:
                    to_del.append(tid)
            for tid in to_del:
                del self.tracks[tid]
            return self.tracks

        # Compute distance matrix
        dists = np.linalg.norm(track_centroids[:, None, :] - det_centroids[None, :, :], axis=2)

        # Greedy assignment: smallest distance pairs first
        pairs = []
        for i, tid in enumerate(track_ids):
            for j in range(len(normalized_dets)):
                pairs.append((float(dists[i, j]), i, j, tid))
        pairs.sort(key=lambda x: x[0])

        assigned_tracks = set()
        assigned_dets = set()

        for dist, i, j, tid in pairs:
            if dist > self.max_distance:
                continue
            if tid in assigned_tracks or j in assigned_dets:
                continue
            assigned_tracks.add(tid)
            assigned_dets.add(j)

            c, b, col = normalized_dets[j]
            tr = self.tracks[tid]
            tr.centroid = c
            tr.bbox = b
            tr.missing = 0
            if col != "unknown":
                tr.color = col

        # Tracks not assigned -> missing
        to_del = []
        for tid in track_ids:
            if tid not in assigned_tracks:
                self.tracks[tid].missing += 1
                if self.tracks[tid].missing > self.max_missing:
                    to_del.append(tid)
        for tid in to_del:
            del self.tracks[tid]

        # Detections not assigned -> new tracks
        for j in range(len(normalized_dets)):
            if j not in assigned_dets:
                c, b, col = normalized_dets[j]
                self._new_track(c, b, col)

        return self.tracks


class LineCrossingCounter:
    def __init__(self):
        self.total = 0
        self.counts = {"Red": 0, "Yellow": 0, "Green": 0, "Blue": 0, "unknown": 0}

    def reset(self) -> None:
        self.total = 0
        self.counts = {"Red": 0, "Yellow": 0, "Green": 0, "Blue": 0, "unknown": 0}

    def update_counts(self, tracks: dict[int, Track], prev_centroids: dict[int, Point], line: Line2D) -> int:
        """Update total count based on line crossing.

        Each track is counted at most once.
        """
        for tid, tr in tracks.items():
            if tr.counted:
                continue
            if tid not in prev_centroids:
                continue
            prev = prev_centroids[tid]
            cur = tr.centroid
            if crossed_line(prev, cur, line):
                tr.counted = True
                self.total += 1
                if tr.color not in self.counts:
                    self.counts[tr.color] = 0
                self.counts[tr.color] += 1
        return self.total
