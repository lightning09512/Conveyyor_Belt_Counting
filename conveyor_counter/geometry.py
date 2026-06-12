from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Line2D:
    """Line represented by two points (x1,y1)-(x2,y2)."""

    x1: float
    y1: float
    x2: float
    y2: float

    def side_value(self, p: Point) -> float:
        """Signed area *2 of triangle (p1->p2->p).

        Equivalent to cross((p2-p1), (p-p1)).
        Sign tells which side of the directed line the point lies on.
        """
        return (self.x2 - self.x1) * (p.y - self.y1) - (self.y2 - self.y1) * (p.x - self.x1)


def crossed_line(prev: Point, cur: Point, line: Line2D) -> bool:
    """Return True if segment prev->cur crosses the infinite line.

    We treat touching (side==0) as crossing to be robust.
    """
    s1 = line.side_value(prev)
    s2 = line.side_value(cur)

    # Same side and not touching => no crossing
    if s1 > 0 and s2 > 0:
        return False
    if s1 < 0 and s2 < 0:
        return False

    return True
