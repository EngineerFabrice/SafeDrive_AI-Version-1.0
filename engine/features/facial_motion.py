"""Frame-to-frame facial and head movement.

Movement is normalised by face size (so it does not depend on the driver's
distance to the camera) and by elapsed time (so it does not depend on the
frame rate). The values describe motion between two consecutive valid frames
only. Judging steadiness, sway or instability over time is left to the later
temporal stage.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..detectors import BoundingBox
from ..detectors.landmarks import FaceLandmarks
from .head_pose import HeadPoseFeatures


@dataclass
class FacialMotionConfig:
    max_gap: float = 0.5   # seconds; a longer gap breaks the motion chain


@dataclass(frozen=True)
class FacialMotionFeatures:
    valid: bool
    dt: float = math.nan                    # seconds since the previous valid frame
    face_speed: float = math.nan            # face-centre speed, face widths per second
    face_dx: float = math.nan               # signed face-centre displacement, face widths
    face_dy: float = math.nan
    scale_change: float = math.nan          # log(width / previous width) per second
    landmark_speed: float = math.nan        # mean landmark speed, interocular distances per second
    landmark_deformation: float = math.nan  # landmark motion left after removing translation, IOD/s
    pitch_rate: float = math.nan            # degrees per second (needs valid head pose in both frames)
    yaw_rate: float = math.nan
    roll_rate: float = math.nan
    reason: str = ""


@dataclass
class _Previous:
    t: float
    face: BoundingBox
    points: Optional[np.ndarray]
    iod: float
    pose: Optional[HeadPoseFeatures]


class FacialMotionTracker:
    def __init__(self, config: Optional[FacialMotionConfig] = None):
        self.config = config or FacialMotionConfig()
        self._prev: Optional[_Previous] = None

    def reset(self) -> None:
        self._prev = None

    def update(self, t: float, face: BoundingBox, landmarks: Optional[FaceLandmarks],
               pose: Optional[HeadPoseFeatures]) -> FacialMotionFeatures:
        current = _Previous(t=t, face=face,
                            points=landmarks.points if landmarks is not None else None,
                            iod=landmarks.interocular if landmarks is not None else math.nan,
                            pose=pose if pose is not None and pose.valid else None)
        prev, self._prev = self._prev, current

        if prev is None:
            return FacialMotionFeatures(valid=False, reason="no previous frame")
        dt = t - prev.t
        if dt <= 0 or dt > self.config.max_gap:
            return FacialMotionFeatures(valid=False, dt=dt, reason="gap since previous frame")

        width = (face.width + prev.face.width) / 2
        if current.points is not None and prev.points is not None:
            # The landmark centroid is more precise than the face box centre.
            (cx, cy), (px, py) = current.points.mean(axis=0), prev.points.mean(axis=0)
        else:
            (cx, cy), (px, py) = face.center, prev.face.center
        face_dx, face_dy = (cx - px) / width, (cy - py) / width
        values = dict(
            dt=dt, face_dx=face_dx, face_dy=face_dy,
            face_speed=math.hypot(face_dx, face_dy) / dt,
            scale_change=math.log(max(face.width, 1) / max(prev.face.width, 1)) / dt,
        )

        if current.points is not None and prev.points is not None:
            iod = (current.iod + prev.iod) / 2
            delta = (current.points - prev.points) / iod
            values["landmark_speed"] = float(np.linalg.norm(delta, axis=1).mean() / dt)
            residual = delta - delta.mean(axis=0)
            values["landmark_deformation"] = float(np.linalg.norm(residual, axis=1).mean() / dt)

        if current.pose is not None and prev.pose is not None:
            values["pitch_rate"] = (current.pose.pitch - prev.pose.pitch) / dt
            values["yaw_rate"] = (current.pose.yaw - prev.pose.yaw) / dt
            roll_delta = (current.pose.roll - prev.pose.roll + 180) % 360 - 180  # wrap at +/-180
            values["roll_rate"] = roll_delta / dt

        return FacialMotionFeatures(valid=True, **values)
