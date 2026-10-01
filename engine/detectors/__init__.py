"""Detectors used by the monitoring pipeline.

Shared geometry lives here; each detector lives in its own module and is
imported explicitly (e.g. ``from engine.detectors.person import PersonDetector``).
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned box in pixel coordinates; (x1, y1) inclusive, (x2, y2) exclusive."""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    def clip(self, frame_width: int, frame_height: int) -> "BoundingBox":
        return BoundingBox(
            max(0, min(self.x1, frame_width)), max(0, min(self.y1, frame_height)),
            max(0, min(self.x2, frame_width)), max(0, min(self.y2, frame_height)),
        )

    def expand(self, fraction: float) -> "BoundingBox":
        dx, dy = int(self.width * fraction), int(self.height * fraction)
        return BoundingBox(self.x1 - dx, self.y1 - dy, self.x2 + dx, self.y2 + dy)

    def iou(self, other: "BoundingBox") -> float:
        inter = BoundingBox(max(self.x1, other.x1), max(self.y1, other.y1),
                            min(self.x2, other.x2), min(self.y2, other.y2)).area
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2
