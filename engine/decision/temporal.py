"""Temporal decision engine: per-frame model outputs -> one stable sobriety assessment.

A single frame never decides. The engine keeps a SHORT sliding window of recent
observations (default: at most 15 frames and 3 seconds) and turns it into one of

    SOBER | UNCERTAIN | POTENTIALLY_NOT_SOBER     (ASSESSING until the first decision)

How the window is scored
------------------------
Each observation is the model's probability that the frame shows a potentially
not-sober driver (``p_not_sober``), the face-quality score of that frame, and a
timestamp. Frames with no prediction or with quality below ``min_quality`` are
counted but are *invalid*: they never contribute to the score.

    temporal_score = sum(w_i * p_i) / sum(w_i)
    w_i = quality_i * confidence_i * 0.5 ** (age_i / recency_half_life_frames)

``confidence_i = max(p_i, 1 - p_i)`` so confident frames weigh more, and the
recency term lets a consistent recent trend outweigh older frames without
letting a single new frame dominate (temporal smoothing).

How the score becomes an assessment
-----------------------------------
* not enough valid frames, too many invalid frames, or low mean model
  confidence  -> UNCERTAIN (with the reason), never a confident result
* score >= ``not_sober_threshold``                 -> POTENTIALLY_NOT_SOBER
* score <= ``sober_threshold``                      -> SOBER
* otherwise                                         -> UNCERTAIN

Hysteresis (stability)
----------------------
* Escalation to POTENTIALLY_NOT_SOBER is immediate (``escalate_confirmations``,
  default 1): the window has already smoothed the evidence and a missed warning
  is the safety-critical error.
* Switching between SOBER and UNCERTAIN needs ``switch_confirmations``
  consecutive evaluations, so the label does not flicker.
* Leaving POTENTIALLY_NOT_SOBER needs the score to fall to ``release_threshold``
  (lower than the entry threshold) for ``release_confirmations`` consecutive
  evaluations AND at least ``min_hold_seconds`` in the state. Poor-quality or
  missing frames never release it: covering the camera must not clear a warning.

The decision is available as soon as ``min_valid_frames`` valid frames exist
(about half a second at 10 FPS with the defaults), and the engine keeps
evaluating every new frame afterwards (continuous lightweight monitoring).
How often frames are fed (inference stride) is decided by the caller.

This module has no model, camera or Flask dependency, so it is tested with
scripted prediction sequences.
"""

import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Deque, Optional, Tuple


class Assessment(str, Enum):
    ASSESSING = "ASSESSING"                       # initial window not ready yet
    SOBER = "SOBER"
    UNCERTAIN = "UNCERTAIN"
    POTENTIALLY_NOT_SOBER = "POTENTIALLY_NOT_SOBER"


class FrameLabel(str, Enum):
    """Frame-level classes, for callers that have a label + confidence instead of a probability."""
    SOBER = "SOBER"
    NOT_SOBER = "NOT_SOBER"


# Reasons attached to an UNCERTAIN result (also shown to researchers).
REASON_INSUFFICIENT_VALID = "insufficient_valid_frames"
REASON_UNSTABLE_FACE = "too_many_invalid_frames"
REASON_LOW_CONFIDENCE = "low_model_confidence"
REASON_AMBIGUOUS = "ambiguous_temporal_score"


@dataclass
class TemporalConfig:
    window_size: int = 15                  # max observations kept
    max_window_seconds: float = 3.0        # observations older than this are dropped
    min_valid_frames: int = 5              # valid frames needed for a prediction-based decision
    min_valid_fraction: float = 0.5        # below this share of valid frames -> UNCERTAIN
    min_quality: float = 0.5               # frames below this face quality are invalid
    min_mean_confidence: float = 0.6       # mean max(p, 1-p) of valid frames
    recency_half_life_frames: float = 4.0  # weight halves every N frames back
    not_sober_threshold: float = 0.65      # enter POTENTIALLY_NOT_SOBER
    sober_threshold: float = 0.35          # score at or below -> SOBER
    release_threshold: float = 0.45        # leave POTENTIALLY_NOT_SOBER only at or below this
    escalate_confirmations: int = 1
    switch_confirmations: int = 2
    release_confirmations: int = 5
    min_hold_seconds: float = 5.0          # minimum time in POTENTIALLY_NOT_SOBER

    def __post_init__(self):
        if not 0.0 <= self.sober_threshold < self.release_threshold < self.not_sober_threshold <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= sober < release < not_sober <= 1")
        if self.min_valid_frames < 1 or self.window_size < self.min_valid_frames:
            raise ValueError("window_size must be >= min_valid_frames >= 1")
        for name in ("escalate_confirmations", "switch_confirmations", "release_confirmations"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")


@dataclass(frozen=True)
class Observation:
    p_not_sober: Optional[float]   # None: no usable prediction for this frame
    quality: float
    timestamp: float
    valid: bool


@dataclass(frozen=True)
class TemporalDecision:
    assessment: Assessment          # stable, user-facing result (after hysteresis)
    raw_assessment: Assessment      # what the current window alone says
    score: Optional[float]          # weighted P(potentially not sober) of the window, 0..1
    confidence: Optional[float]     # confidence in `assessment`; None for ASSESSING/UNCERTAIN
    valid_frames: int
    total_frames: int
    reasons: Tuple[str, ...]
    changed: bool                   # True when `assessment` differs from the previous decision
    timestamp: float

    def to_dict(self) -> dict:
        data = asdict(self)
        data["assessment"] = self.assessment.value
        data["raw_assessment"] = self.raw_assessment.value
        data["reasons"] = list(self.reasons)
        for key in ("score", "confidence"):
            if data[key] is not None:
                data[key] = round(data[key], 4)
        return data


class TemporalDecisionEngine:
    def __init__(self, config: Optional[TemporalConfig] = None):
        self.config = config or TemporalConfig()
        self._window: Deque[Observation] = deque(maxlen=self.config.window_size)
        self.reset()

    # ------------------------------------------------------------ input
    def add_prediction(self, p_not_sober: Optional[float] = None, quality: float = 1.0,
                       timestamp: Optional[float] = None, label: Optional[str] = None,
                       confidence: Optional[float] = None) -> Observation:
        """Add one frame. Give either ``p_not_sober`` or ``label`` + ``confidence``.

        ``label="NOT_SOBER", confidence=0.9`` equals ``p_not_sober=0.9``;
        ``label="SOBER", confidence=0.9`` equals ``p_not_sober=0.1``.
        Pass neither for a frame without a usable prediction (no face, model unavailable).
        """
        if label is not None:
            if p_not_sober is not None:
                raise ValueError("give p_not_sober or label/confidence, not both")
            label = FrameLabel(label)
            c = 1.0 if confidence is None else float(confidence)
            if not 0.0 <= c <= 1.0:
                raise ValueError("confidence must be in [0, 1]")
            p_not_sober = c if label == FrameLabel.NOT_SOBER else 1.0 - c
        if p_not_sober is not None:
            p_not_sober = float(p_not_sober)
            if math.isnan(p_not_sober) or not 0.0 <= p_not_sober <= 1.0:
                p_not_sober = None   # malformed model output: treat as a missing prediction
        quality = float(quality) if quality is not None and not math.isnan(quality) else 0.0
        ts = time.monotonic() if timestamp is None else float(timestamp)

        obs = Observation(p_not_sober=p_not_sober, quality=quality, timestamp=ts,
                          valid=p_not_sober is not None and quality >= self.config.min_quality)
        self._window.append(obs)
        self._prune(ts)
        return obs

    def update(self, **kwargs) -> TemporalDecision:
        """``add_prediction(**kwargs)`` followed by ``make_decision()``."""
        self.add_prediction(**kwargs)
        return self.make_decision()

    # ------------------------------------------------------------ state
    def reset(self) -> None:
        """Forget all observations and decisions (new driver / new monitoring session)."""
        self._window.clear()
        self._state = Assessment.ASSESSING
        self._pending: Optional[Assessment] = None
        self._pending_count = 0
        self._release_count = 0
        self._entered_at: Optional[float] = None
        self._last: Optional[TemporalDecision] = None

    @property
    def assessment(self) -> Assessment:
        return self._state

    @property
    def last_decision(self) -> Optional[TemporalDecision]:
        return self._last

    def _prune(self, now: float) -> None:
        while self._window and now - self._window[0].timestamp > self.config.max_window_seconds:
            self._window.popleft()

    def _valid(self):
        return [o for o in self._window if o.valid]

    def is_window_ready(self) -> bool:
        """True when a decision can be made: enough valid frames, or a full window of mostly invalid ones."""
        return (len(self._valid()) >= self.config.min_valid_frames
                or len(self._window) >= self.config.window_size)

    # ------------------------------------------------------------ scoring
    def calculate_temporal_score(self) -> Optional[float]:
        """Confidence-, quality- and recency-weighted mean P(not sober); None without valid frames."""
        valid = self._valid()
        if not valid:
            return None
        n = len(valid)
        num = den = 0.0
        for i, o in enumerate(valid):
            age = n - 1 - i                                     # 0 for the newest frame
            w = o.quality * max(o.p_not_sober, 1.0 - o.p_not_sober) \
                * 0.5 ** (age / self.config.recency_half_life_frames)
            num += w * o.p_not_sober
            den += w
        return num / den if den > 0 else None

    def _raw(self, score: Optional[float]) -> Tuple[Assessment, Tuple[str, ...]]:
        cfg = self.config
        valid = self._valid()
        if len(valid) < cfg.min_valid_frames:
            return Assessment.UNCERTAIN, (REASON_INSUFFICIENT_VALID,)
        if len(valid) / len(self._window) < cfg.min_valid_fraction:
            return Assessment.UNCERTAIN, (REASON_UNSTABLE_FACE,)
        mean_conf = sum(max(o.p_not_sober, 1 - o.p_not_sober) for o in valid) / len(valid)
        if mean_conf < cfg.min_mean_confidence:
            return Assessment.UNCERTAIN, (REASON_LOW_CONFIDENCE,)
        if score >= cfg.not_sober_threshold:
            return Assessment.POTENTIALLY_NOT_SOBER, ()
        if score <= cfg.sober_threshold:
            return Assessment.SOBER, ()
        return Assessment.UNCERTAIN, (REASON_AMBIGUOUS,)

    # ------------------------------------------------------------ decision
    def make_decision(self) -> TemporalDecision:
        cfg = self.config
        now = self._window[-1].timestamp if self._window else time.monotonic()
        score = self.calculate_temporal_score()
        previous = self._state

        if not self.is_window_ready():
            raw, reasons = Assessment.ASSESSING, ()
        else:
            raw, reasons = self._raw(score)
            self._apply_hysteresis(raw, reasons, score, now)

        state = self._state
        if state == Assessment.POTENTIALLY_NOT_SOBER:
            confidence = score
        elif state == Assessment.SOBER and score is not None:
            confidence = 1.0 - score
        else:
            confidence = None
        decision = TemporalDecision(
            assessment=state, raw_assessment=raw, score=score, confidence=confidence,
            valid_frames=len(self._valid()), total_frames=len(self._window),
            reasons=reasons if state == Assessment.UNCERTAIN or raw == Assessment.UNCERTAIN else (),
            changed=state != previous, timestamp=now)
        self._last = decision
        return decision

    def _apply_hysteresis(self, raw: Assessment, reasons: Tuple[str, ...], score: Optional[float],
                          now: float) -> None:
        cfg = self.config
        state = self._state

        if state == Assessment.ASSESSING:            # first decision: no confirmation needed
            self._enter(raw, now)
            return

        if state == Assessment.POTENTIALLY_NOT_SOBER:
            # Only consistent, usable sober-leaning evidence can release the warning. Missing,
            # low-quality or low-confidence data (UNCERTAIN for a data reason) never does.
            usable = raw == Assessment.SOBER or reasons == (REASON_AMBIGUOUS,)
            evidence_ok = usable and score is not None and score <= cfg.release_threshold
            self._release_count = self._release_count + 1 if evidence_ok else 0
            held = now - (self._entered_at or now)
            if self._release_count >= cfg.release_confirmations and held >= cfg.min_hold_seconds:
                self._enter(raw, now)
            return

        if raw == state:
            self._pending, self._pending_count = None, 0
            return
        needed = cfg.escalate_confirmations if raw == Assessment.POTENTIALLY_NOT_SOBER \
            else cfg.switch_confirmations
        if raw == self._pending:
            self._pending_count += 1
        else:
            self._pending, self._pending_count = raw, 1
        if self._pending_count >= needed:
            self._enter(raw, now)

    def _enter(self, state: Assessment, now: float) -> None:
        if state != self._state:
            self._entered_at = now
        self._state = state
        self._pending, self._pending_count = None, 0
        self._release_count = 0
