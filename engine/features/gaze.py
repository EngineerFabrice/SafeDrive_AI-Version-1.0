"""Eye-in-head gaze proxies from the eye regions.

With five landmarks there are no eye-corner or iris points, so gaze is
estimated from image content around each eye centre:

* ``horizontal``: sclera balance. It compares the bright (white-of-eye)
  pixels on each side of the dark iris along the eye line, giving a value in
  [-1, 1]: 0 means centred, positive means the iris sits towards the image
  right. This does not depend on where exactly the landmark sits.
* ``vertical``: offset of the dark-region centroid perpendicular to the eye
  line, normalised by eye-patch half-height. Less reliable, because the
  eyelids and lashes are also dark.

These are uncalibrated proxies, not gaze angles. A single frame says nothing
about impairment. Gaze instability (for example nystagmus-like oscillation)
can only be judged over time by a later temporal stage, which also needs to
combine these values with head pose. No eye-openness or blink measure is
produced here.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from ..detectors.landmarks import FaceLandmarks


@dataclass
class GazeConfig:
    patch_width: float = 0.50       # eye patch size relative to interocular distance
    patch_height: float = 0.24
    min_patch_px: int = 12          # minimum eye patch width in pixels
    dark_percentile: float = 15.0   # pixels darker than this are treated as iris/pupil
    bright_percentile: float = 80.0  # pixels brighter than this are treated as sclera
    min_contrast: float = 0.12      # (bright - dark) / 255 needed for a usable eye


@dataclass(frozen=True)
class GazeFeatures:
    valid: bool
    horizontal: float = math.nan       # mean sclera balance of both eyes, [-1, 1]
    vertical: float = math.nan         # mean normalised iris offset across the eye line
    horizontal_left: float = math.nan  # eye on the image left
    horizontal_right: float = math.nan
    eye_disagreement: float = math.nan  # |left - right| horizontal; large = unreliable frame
    eye_contrast: float = math.nan     # mean iris/sclera contrast (quality)
    eyes_used: int = 0
    reason: str = ""


def _eye(gray: np.ndarray, centre: np.ndarray, axis: np.ndarray, iod: float,
         cfg: GazeConfig) -> Optional[Tuple[float, float, float]]:
    """(horizontal balance, vertical offset, contrast) for one eye, or None."""
    half_w, half_h = int(cfg.patch_width * iod / 2), int(cfg.patch_height * iod / 2)
    if half_w * 2 < cfg.min_patch_px or half_h < 2:
        return None
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    y1, y2, x1, x2 = cy - half_h, cy + half_h, cx - half_w, cx + half_w
    if x1 < 0 or y1 < 0 or x2 > gray.shape[1] or y2 > gray.shape[0]:
        return None
    patch = cv2.GaussianBlur(gray[y1:y2, x1:x2], (3, 3), 0).astype(np.float32)

    dark_t, bright_t = np.percentile(patch, (cfg.dark_percentile, cfg.bright_percentile))
    contrast = float((bright_t - dark_t) / 255.0)
    if contrast < cfg.min_contrast:
        return None

    ys, xs = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
    dx, dy = xs - (patch.shape[1] - 1) / 2, ys - (patch.shape[0] - 1) / 2
    along = dx * axis[0] + dy * axis[1]           # coordinate along the eye line
    across = -dx * axis[1] + dy * axis[0]         # perpendicular to it (down = positive)

    dark = patch <= dark_t
    weights = (dark_t - patch[dark]) + 1.0
    iris_along = float((along[dark] * weights).sum() / weights.sum())
    iris_across = float((across[dark] * weights).sum() / weights.sum())

    bright = patch >= bright_t
    left = np.count_nonzero(bright & (along < iris_along))
    right = np.count_nonzero(bright & (along > iris_along))
    if left + right == 0:
        return None
    # More white on the left means the iris has moved to the right.
    balance = (left - right) / (left + right)
    return balance, iris_across / half_h, contrast


class GazeEstimator:
    def __init__(self, config: Optional[GazeConfig] = None):
        self.config = config or GazeConfig()

    def estimate(self, gray: np.ndarray, offset: Tuple[int, int],
                 landmarks: Optional[FaceLandmarks]) -> GazeFeatures:
        """``gray`` is a grayscale crop whose top-left corner is at ``offset`` in the frame."""
        if landmarks is None:
            return GazeFeatures(valid=False, reason="no landmarks")
        iod = landmarks.interocular
        if iod <= 0:
            return GazeFeatures(valid=False, reason="degenerate landmarks")
        axis = (landmarks.eye_right - landmarks.eye_left) / iod

        origin = np.asarray(offset, np.float32)
        eyes = [_eye(gray, e - origin, axis, iod, self.config)
                for e in (landmarks.eye_left, landmarks.eye_right)]
        usable = [e for e in eyes if e is not None]
        if not usable:
            return GazeFeatures(valid=False, reason="eye regions too small or low contrast")

        h_left = eyes[0][0] if eyes[0] else math.nan
        h_right = eyes[1][0] if eyes[1] else math.nan
        return GazeFeatures(
            valid=True,
            horizontal=float(np.mean([e[0] for e in usable])),
            vertical=float(np.mean([e[1] for e in usable])),
            horizontal_left=h_left, horizontal_right=h_right,
            eye_disagreement=abs(h_left - h_right) if len(usable) == 2 else math.nan,
            eye_contrast=float(np.mean([e[2] for e in usable])),
            eyes_used=len(usable))
