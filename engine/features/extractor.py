"""Orchestrates per-frame feature extraction for the detected driver face.

face (Phase 1) -> landmarks -> appearance | head pose | gaze | facial motion
               -> FaceFeatures (structured, plus a fixed-order numeric vector)

The output only says whether features are available. It never makes an
impairment decision. Each ``FaceFeatures`` carries a timestamp and frame id,
and ``to_vector()`` always has the same length and order (``FEATURE_NAMES``,
NaN for anything unavailable), so a later temporal stage can keep a
t1, t2, t3, ... sequence without knowing about the individual modules.
"""

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import cv2
import numpy as np

from ..detectors.face import FaceDetection
from ..detectors.landmarks import FaceLandmarks, LandmarkDetector, LandmarkDetectorConfig
from ..state import FeatureStatus
from .appearance import AppearanceConfig, AppearanceExtractor, AppearanceFeatures
from .facial_motion import FacialMotionConfig, FacialMotionFeatures, FacialMotionTracker
from .gaze import GazeConfig, GazeEstimator, GazeFeatures
from .head_pose import HeadPoseConfig, HeadPoseEstimator, HeadPoseFeatures


@dataclass
class FeatureExtractorConfig:
    landmarks: LandmarkDetectorConfig = field(default_factory=LandmarkDetectorConfig)
    appearance: AppearanceConfig = field(default_factory=AppearanceConfig)
    head_pose: HeadPoseConfig = field(default_factory=HeadPoseConfig)
    gaze: GazeConfig = field(default_factory=GazeConfig)
    motion: FacialMotionConfig = field(default_factory=FacialMotionConfig)


# (group, attribute) pairs that make up the numeric vector, in a fixed order.
_VECTOR_LAYOUT = (
    ("head_pose", ("pitch", "yaw", "roll", "yaw_ratio", "pitch_ratio", "roll_2d",
                   "reprojection_error", "confidence")),
    ("gaze", ("horizontal", "vertical", "horizontal_left", "horizontal_right",
              "eye_disagreement", "eye_contrast")),
    ("appearance", ("skin_a", "skin_red_ratio", "eye_region_a_rel", "brightness",
                    "contrast", "sharpness")),
    ("facial_motion", ("face_speed", "face_dx", "face_dy", "scale_change", "landmark_speed",
                       "landmark_deformation", "pitch_rate", "yaw_rate", "roll_rate")),
)
FEATURE_NAMES = tuple(f"{group}.{name}" for group, names in _VECTOR_LAYOUT for name in names)
# Bump whenever FEATURE_NAMES changes (names or order); documented in features/README.md.
FEATURE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FaceFeatures:
    status: FeatureStatus
    timestamp: float                   # wall-clock (epoch seconds) of the source frame
    frame_id: Optional[int] = None
    landmark_score: float = math.nan
    head_pose: HeadPoseFeatures = HeadPoseFeatures(valid=False)
    gaze: GazeFeatures = GazeFeatures(valid=False)
    appearance: AppearanceFeatures = AppearanceFeatures(valid=False)
    facial_motion: FacialMotionFeatures = FacialMotionFeatures(valid=False)
    processing_time: float = 0.0       # ms
    reason: str = ""

    @property
    def valid(self) -> bool:
        return self.status == FeatureStatus.FEATURES_AVAILABLE

    def to_vector(self) -> np.ndarray:
        """Fixed-order float32 vector matching ``FEATURE_NAMES``; invalid groups are NaN."""
        values = []
        for group, names in _VECTOR_LAYOUT:
            part = getattr(self, group)
            values.extend(getattr(part, n) if part.valid else math.nan for n in names)
        return np.asarray(values, dtype=np.float32)

    def to_dict(self) -> dict:
        """JSON-safe dict: NaN becomes None, floats rounded."""
        def clean(value):
            if isinstance(value, np.integer):
                return int(value)
            if isinstance(value, (float, np.floating)):
                return None if math.isnan(value) else round(float(value), 4)
            if isinstance(value, dict):
                return {k: clean(v) for k, v in value.items()}
            return value

        data = clean(asdict(self))
        data["status"] = self.status.value
        data["valid"] = self.valid
        data["schema_version"] = FEATURE_SCHEMA_VERSION
        return data

    @classmethod
    def unavailable(cls, reason: str, timestamp: Optional[float] = None,
                    frame_id: Optional[int] = None) -> "FaceFeatures":
        return cls(status=FeatureStatus.FEATURES_UNAVAILABLE,
                   timestamp=time.time() if timestamp is None else timestamp,
                   frame_id=frame_id, reason=reason)


class FeatureExtractor:
    def __init__(self, config: Optional[FeatureExtractorConfig] = None,
                 landmark_detector: Optional[LandmarkDetector] = None):
        self.config = config or FeatureExtractorConfig()
        self.landmarks = landmark_detector or LandmarkDetector(self.config.landmarks)
        self.appearance = AppearanceExtractor(self.config.appearance)
        self.head_pose = HeadPoseEstimator(self.config.head_pose)
        self.gaze = GazeEstimator(self.config.gaze)
        self.motion = FacialMotionTracker(self.config.motion)

    def load(self) -> bool:
        return self.landmarks.load()

    def reset(self) -> None:
        """Call when the face is lost so motion is not measured across the gap."""
        self.motion.reset()

    def extract(self, frame: np.ndarray, face: FaceDetection, captured_at: float,
                timestamp: Optional[float] = None, frame_id: Optional[int] = None) -> FaceFeatures:
        """``captured_at`` is a monotonic time (perf_counter) used for motion rates."""
        t0 = time.perf_counter()
        timestamp = time.time() if timestamp is None else timestamp

        def done(**kw) -> FaceFeatures:
            return FaceFeatures(timestamp=timestamp, frame_id=frame_id,
                                processing_time=(time.perf_counter() - t0) * 1000, **kw)

        # Reuse landmarks from the face detector when it produced them (YuNet backend);
        # only run the landmark model separately for a Haar fallback box.
        lm: Optional[FaceLandmarks] = face.landmarks
        if lm is None:
            if not self.landmarks.load():
                self.motion.reset()
                return done(status=FeatureStatus.FEATURES_UNAVAILABLE,
                            reason=f"landmark model unavailable: {self.landmarks.error}")
            lm = self.landmarks.detect(frame, face.bbox)
            if lm is None:
                self.motion.reset()
                return done(status=FeatureStatus.FEATURES_UNAVAILABLE, reason="facial landmarks not found")

        h, w = frame.shape[:2]
        region = lm.bbox.expand(0.1).clip(w, h)   # grayscale only the face, not the whole frame
        gray = cv2.cvtColor(frame[region.y1:region.y2, region.x1:region.x2], cv2.COLOR_BGR2GRAY)
        head_pose = self.head_pose.estimate(lm, frame.shape)
        gaze = self.gaze.estimate(gray, (region.x1, region.y1), lm)
        appearance = self.appearance.extract(frame, face.bbox, lm)
        # YuNet's box is more stable frame to frame than the Haar box, so use it for motion.
        motion = self.motion.update(captured_at, lm.bbox, lm, head_pose)

        # Head pose and appearance are the core per-frame features. Gaze and
        # motion may be missing for single frames (low eye contrast, first frame)
        # and are then NaN in the vector.
        core_ok = head_pose.valid and appearance.valid
        reasons = [f"{name}: {part.reason}" for name, part in
                   (("head_pose", head_pose), ("appearance", appearance)) if not part.valid]
        return done(
            status=FeatureStatus.FEATURES_AVAILABLE if core_ok else FeatureStatus.FEATURES_UNAVAILABLE,
            landmark_score=lm.score, head_pose=head_pose, gaze=gaze, appearance=appearance,
            facial_motion=motion, reason="; ".join(reasons))
