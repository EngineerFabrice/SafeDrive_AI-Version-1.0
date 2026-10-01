"""Head pose (pitch / yaw / roll) from five facial landmarks.

Pose is recovered with ``cv2.solvePnP`` (SQPnP) against a generic 3D face
model, with a pinhole camera approximated from the frame size (focal length =
frame width, principal point at the centre). Five points and an
uncalibrated camera give a coarse but stable estimate; the reprojection
error is reported so later stages can weight or reject frames.

Simple 2D landmark ratios are returned as well. They need no camera model
and are robust model inputs even when the 3D fit is poor.

Angle conventions (degrees, from the camera's point of view):
  yaw   > 0  face turned towards the image right
  pitch > 0  face tilted up
  roll  > 0  face rotated clockwise in the image
"""

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ..detectors.landmarks import FaceLandmarks

# Generic face in millimetres, camera-aligned axes (x right, y down, z away
# from the camera), nose tip at the origin. Same order as FaceLandmarks.points.
_MODEL_POINTS = np.array([
    [-31.5, -32.0, 27.0],   # eye, image left
    [31.5, -32.0, 27.0],    # eye, image right
    [0.0, 0.0, 0.0],        # nose tip
    [-25.0, 30.0, 23.0],    # mouth corner, image left
    [25.0, 30.0, 23.0],     # mouth corner, image right
], dtype=np.float64)


@dataclass
class HeadPoseConfig:
    max_reprojection_error: float = 0.12   # mean error / interocular distance
    min_interocular_px: float = 20.0


@dataclass(frozen=True)
class HeadPoseFeatures:
    valid: bool
    pitch: float = math.nan
    yaw: float = math.nan
    roll: float = math.nan
    reprojection_error: float = math.nan   # mean, normalised by interocular distance
    yaw_ratio: float = math.nan            # nose x offset from eye midpoint / interocular
    pitch_ratio: float = math.nan          # nose height between eye line (0) and mouth line (1)
    roll_2d: float = math.nan              # eye-line angle in the image, degrees
    confidence: float = 0.0                # landmark score x fit quality, 0..1
    reason: str = ""


def _ratios(lm: FaceLandmarks):
    iod = lm.interocular
    eye_mid = (lm.eye_left + lm.eye_right) / 2
    mouth_mid = (lm.mouth_left + lm.mouth_right) / 2
    d = lm.eye_right - lm.eye_left
    roll_2d = math.degrees(math.atan2(d[1], d[0]))
    yaw_ratio = float((lm.nose[0] - eye_mid[0]) / iod)
    span = mouth_mid[1] - eye_mid[1]
    pitch_ratio = float((lm.nose[1] - eye_mid[1]) / span) if abs(span) > 1e-6 else math.nan
    return yaw_ratio, pitch_ratio, roll_2d


class HeadPoseEstimator:
    def __init__(self, config: Optional[HeadPoseConfig] = None):
        self.config = config or HeadPoseConfig()

    def estimate(self, landmarks: Optional[FaceLandmarks], frame_shape) -> HeadPoseFeatures:
        if landmarks is None:
            return HeadPoseFeatures(valid=False, reason="no landmarks")
        iod = landmarks.interocular
        if iod < self.config.min_interocular_px:
            return HeadPoseFeatures(valid=False, reason=f"face too small (interocular {iod:.0f}px)")

        yaw_ratio, pitch_ratio, roll_2d = _ratios(landmarks)

        h, w = frame_shape[:2]
        camera = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
        image_points = landmarks.points.astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(_MODEL_POINTS, image_points, camera, None,
                                      flags=cv2.SOLVEPNP_SQPNP)
        if not ok or tvec[2, 0] <= 0:
            return HeadPoseFeatures(valid=False, yaw_ratio=yaw_ratio, pitch_ratio=pitch_ratio,
                                    roll_2d=roll_2d, reason="pose solver failed")

        projected, _ = cv2.projectPoints(_MODEL_POINTS, rvec, tvec, camera, None)
        error = float(np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1).mean() / iod)

        rot, _ = cv2.Rodrigues(rvec)
        forward = rot @ np.array([0.0, 0.0, -1.0])   # direction the face points (towards camera when frontal)
        right = rot @ np.array([1.0, 0.0, 0.0])
        yaw = math.degrees(math.atan2(forward[0], -forward[2]))
        pitch = math.degrees(math.atan2(-forward[1], math.hypot(forward[0], forward[2])))
        roll = math.degrees(math.atan2(right[1], right[0]))

        valid = error <= self.config.max_reprojection_error
        confidence = landmarks.score * max(0.0, 1.0 - error / self.config.max_reprojection_error)
        return HeadPoseFeatures(
            valid=valid, pitch=pitch, yaw=yaw, roll=roll, reprojection_error=error,
            yaw_ratio=yaw_ratio, pitch_ratio=pitch_ratio, roll_2d=roll_2d,
            confidence=float(confidence),
            reason="" if valid else f"poor model fit (error {error:.2f})")
