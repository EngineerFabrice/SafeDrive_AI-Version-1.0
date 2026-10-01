"""Select the driver among detected persons for a fixed, driver-facing camera.

Reasoning for the heuristic
---------------------------
The camera is mounted facing the driver (dashboard / A-pillar), so:

* The driver is the closest person to the lens and therefore usually the
  *largest* person box. Passengers in the back, or pedestrians seen through
  a window, are smaller.
* The camera is aimed at the driver, so the driver sits near a predictable
  position in the image (``expected_center``; image centre by default,
  configurable if the camera is mounted off-axis, e.g. on the centre console).
* The driver does not change between frames, so a candidate that overlaps
  the previous driver box is preferred, which stops the selection jumping
  to a passenger for a single frame.

Each candidate gets a weighted score from these three cues. Candidates
smaller than ``min_area_fraction`` of the frame are rejected outright: they
cannot be the driver of a driver-facing camera, and accepting them would
turn arbitrary people into "the driver".

The returned ROI is the person box padded slightly so the face is not cut
off at the box edge. Replacing this module with a seat-zone calibration or a
tracker later only requires keeping the ``DriverROISelector.select`` interface.
"""

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from . import BoundingBox
from .person import PersonDetection


@dataclass
class DriverROIConfig:
    min_area_fraction: float = 0.04                 # of the frame area
    expected_center: Tuple[float, float] = (0.5, 0.5)  # normalised (x, y)
    area_weight: float = 0.55
    center_weight: float = 0.30
    tracking_weight: float = 0.15
    roi_padding: float = 0.10                       # fraction of box size added on each side


@dataclass(frozen=True)
class DriverROI:
    person: PersonDetection
    roi: BoundingBox     # padded region to search for the face
    score: float


class DriverROISelector:
    def __init__(self, config: Optional[DriverROIConfig] = None):
        self.config = config or DriverROIConfig()
        self._previous: Optional[BoundingBox] = None

    def reset(self) -> None:
        self._previous = None

    def select(self, persons: Sequence[PersonDetection], frame_shape) -> Optional[DriverROI]:
        h, w = frame_shape[:2]
        frame_area = float(w * h)
        cfg = self.config
        max_dist = math.hypot(1.0, 1.0)

        best: Optional[DriverROI] = None
        for person in persons:
            area_frac = person.bbox.area / frame_area
            if area_frac < cfg.min_area_fraction:
                continue
            cx, cy = person.bbox.center
            dist = math.hypot(cx / w - cfg.expected_center[0], cy / h - cfg.expected_center[1])
            center_score = 1.0 - dist / max_dist
            area_score = min(1.0, area_frac / 0.5)  # saturates once the person fills half the frame
            track_score = person.bbox.iou(self._previous) if self._previous else 0.0
            score = (cfg.area_weight * area_score + cfg.center_weight * center_score
                     + cfg.tracking_weight * track_score)
            if best is None or score > best.score:
                roi = person.bbox.expand(cfg.roi_padding).clip(w, h)
                best = DriverROI(person=person, roi=roi, score=score)

        self._previous = best.person.bbox if best else None
        return best
