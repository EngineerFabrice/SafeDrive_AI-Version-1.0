"""Per-frame facial appearance measurements.

These are colour and image-quality measurements, not judgements about the
person. The colour cues reflect visible signs that observational alcohol
checks look for, such as facial flushing (vasodilation) and reddened eyes.
They are also strongly affected by lighting, skin tone, camera white balance
and many non-alcohol causes, so none of them indicates intoxication on its
own. They are inputs for a trained model, ideally used relative to the same
driver's own baseline, and are always reported next to the quality measures
(brightness, contrast, sharpness) the model needs to discount bad frames.

Colour is measured in CIELAB, where a* is the green-red axis and is less
sensitive to brightness changes than raw RGB.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..detectors import BoundingBox
from ..detectors.landmarks import FaceLandmarks


@dataclass
class AppearanceConfig:
    min_patch_px: int = 6               # skin / eye patches smaller than this are unusable
    # Mean L* (0-100) below which colour is meaningless (near-black). Kept low on
    # purpose: a higher cut-off would systematically drop darker-skinned or
    # backlit drivers. Brightness is in the output so the model can weight it.
    min_brightness: float = 5.0
    min_eye_interocular_px: float = 40.0  # eye-region colour needs enough eye resolution
    sharpness_width: int = 128          # face is resized to this width before measuring sharpness


@dataclass(frozen=True)
class AppearanceFeatures:
    valid: bool
    skin_a: float = math.nan              # mean a* of cheek skin (0 = neutral, + = redder)
    skin_red_ratio: float = math.nan      # mean R / (R + G + B) of cheek skin
    eye_region_a_rel: float = math.nan    # eye-region a* minus skin a* (needs landmarks)
    brightness: float = math.nan          # mean L* of the face (0-100)
    contrast: float = math.nan            # std of L* across the face
    sharpness: float = math.nan           # variance of Laplacian at a fixed scale
    region_source: str = ""               # "landmarks" or "face_box"
    reason: str = ""


Patch = Tuple[int, int, int]  # centre x, centre y, half size (face-crop coordinates)


def _cheek_patches(face: BoundingBox, lm: Optional[FaceLandmarks]) -> Tuple[List[Patch], str]:
    if lm is not None:
        iod = lm.interocular
        half = max(1, int(0.10 * iod))
        patches = []
        for eye, mouth, side in ((lm.eye_left, lm.mouth_left, -1), (lm.eye_right, lm.mouth_right, 1)):
            centre = eye + 0.55 * (mouth - eye)          # below the eye, above the mouth corner
            centre[0] += side * 0.12 * iod               # move outwards, away from the nose
            patches.append((int(centre[0]) - face.x1, int(centre[1]) - face.y1, half))
        return patches, "landmarks"
    # No landmarks: fixed proportions of a frontal face box.
    half = max(1, int(0.07 * face.width))
    return [(int(face.width * fx), int(face.height * 0.62), half) for fx in (0.27, 0.73)], "face_box"


def _patch_pixels(img: np.ndarray, patch: Patch, min_px: int) -> Optional[np.ndarray]:
    cx, cy, half = patch
    if half * 2 < min_px:
        return None
    y1, y2, x1, x2 = max(0, cy - half), min(img.shape[0], cy + half), max(0, cx - half), min(img.shape[1], cx + half)
    if y2 - y1 < min_px or x2 - x1 < min_px:
        return None
    return img[y1:y2, x1:x2].reshape(-1, img.shape[2])


class AppearanceExtractor:
    def __init__(self, config: Optional[AppearanceConfig] = None):
        self.config = config or AppearanceConfig()

    def extract(self, frame: np.ndarray, face: BoundingBox,
                landmarks: Optional[FaceLandmarks]) -> AppearanceFeatures:
        cfg = self.config
        crop = frame[face.y1:face.y2, face.x1:face.x2]
        if crop.size == 0:
            return AppearanceFeatures(valid=False, reason="empty face region")

        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        lightness = lab[..., 0] * (100.0 / 255.0)
        brightness, contrast = float(lightness.mean()), float(lightness.std())

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        scale = cfg.sharpness_width / gray.shape[1]
        gray = cv2.resize(gray, (cfg.sharpness_width, max(1, int(gray.shape[0] * scale))))
        sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())

        if brightness < cfg.min_brightness:
            return AppearanceFeatures(valid=False, brightness=brightness, contrast=contrast,
                                      sharpness=sharpness, reason="face too dark")

        patches, source = _cheek_patches(face, landmarks)
        lab_px, bgr_px = [], []
        for patch in patches:
            p = _patch_pixels(lab, patch, cfg.min_patch_px)
            if p is not None:
                lab_px.append(p)
                bgr_px.append(_patch_pixels(crop, patch, cfg.min_patch_px).astype(np.float32))
        if not lab_px:
            return AppearanceFeatures(valid=False, brightness=brightness, contrast=contrast,
                                      sharpness=sharpness, region_source=source,
                                      reason="skin regions too small")

        skin_lab = np.concatenate(lab_px)
        skin_bgr = np.concatenate(bgr_px)
        skin_a = float(skin_lab[:, 1].mean() - 128.0)
        skin_red_ratio = float((skin_bgr[:, 2] / (skin_bgr.sum(axis=1) + 1e-6)).mean())

        eye_a_rel = math.nan
        if landmarks is not None and landmarks.interocular >= cfg.min_eye_interocular_px:
            half = max(1, int(0.12 * landmarks.interocular))
            eye_px = [_patch_pixels(lab, (int(e[0]) - face.x1, int(e[1]) - face.y1, half), cfg.min_patch_px)
                      for e in (landmarks.eye_left, landmarks.eye_right)]
            eye_px = [p for p in eye_px if p is not None]
            if eye_px:
                eye_a_rel = float(np.concatenate(eye_px)[:, 1].mean() - 128.0 - skin_a)

        return AppearanceFeatures(
            valid=True, skin_a=skin_a, skin_red_ratio=skin_red_ratio, eye_region_a_rel=eye_a_rel,
            brightness=brightness, contrast=contrast, sharpness=sharpness, region_source=source)
