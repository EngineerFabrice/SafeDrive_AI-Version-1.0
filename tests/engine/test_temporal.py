"""Temporal Decision Engine with scripted prediction sequences (no model, no camera)."""
import pytest

from engine.decision import Assessment, TemporalConfig, TemporalDecisionEngine
from engine.decision.temporal import (REASON_INSUFFICIENT_VALID, REASON_LOW_CONFIDENCE,
                                      REASON_UNSTABLE_FACE)

S, N = "SOBER", "NOT_SOBER"
FPS = 10.0


def run(sequence, engine=None, confidence=0.9, start=0.0, quality=1.0):
    """Feed labels at FPS; returns the list of decisions (one per frame)."""
    engine = engine or TemporalDecisionEngine()
    out = []
    for i, label in enumerate(sequence):
        ts = start + i / FPS
        if label is None:
            out.append(engine.update(timestamp=ts))                      # no usable prediction
        else:
            out.append(engine.update(label=label, confidence=confidence, quality=quality, timestamp=ts))
    return engine, out


def states(decisions):
    return [d.assessment for d in decisions]


# ---------------------------------------------------------------- spec scenarios
def test_scenario1_consistently_sober_decides_fast():
    _, d = run([S] * 5)
    assert states(d)[:4] == [Assessment.ASSESSING] * 4
    assert d[4].assessment == Assessment.SOBER          # decided after 5 frames (0.4 s at 10 FPS)
    assert d[4].changed and d[4].confidence > 0.85


def test_scenario2_single_noisy_not_sober_does_not_trigger():
    _, d = run([S, S, S, N, S])
    assert d[-1].assessment == Assessment.SOBER
    assert Assessment.POTENTIALLY_NOT_SOBER not in states(d)


def test_scenario3_consistent_impairment_pattern():
    _, d = run([S, N, N, N, N])
    assert d[-1].assessment == Assessment.POTENTIALLY_NOT_SOBER
    assert d[-1].changed and d[-1].confidence >= 0.65


def test_master_example_sequence():
    """S S S N S N N N N: no reaction to the first N, no flip back to SOBER, warning by frame 9."""
    seq = [S, S, S, N, S, N, N, N, N]
    _, d = run(seq)
    st = states(d)
    assert st[3] == Assessment.ASSESSING                 # first NOT SOBER alone never decides
    assert st[4] == Assessment.SOBER                     # initial decision from 5 frames
    assert st[-1] == Assessment.POTENTIALLY_NOT_SOBER    # 0.9 s after start at 10 FPS
    first_pns = st.index(Assessment.POTENTIALLY_NOT_SOBER)
    assert all(s != Assessment.SOBER for s in st[first_pns:])
    # once the trend turned, the label only escalates (no SOBER <-> NOT SOBER ping-pong)
    order = {Assessment.ASSESSING: 0, Assessment.SOBER: 1, Assessment.UNCERTAIN: 2,
             Assessment.POTENTIALLY_NOT_SOBER: 3}
    ranks = [order[s] for s in st]
    assert ranks == sorted(ranks)


def test_alternating_predictions_do_not_flip_labels():
    _, d = run([S, N] * 10)
    st = states(d)
    changes = sum(1 for a, b in zip(st, st[1:]) if a != b)
    assert changes <= 2
    assert Assessment.SOBER not in st[6:] or Assessment.POTENTIALLY_NOT_SOBER not in st[6:]


# ---------------------------------------------------------------- confidence awareness
def test_confidence_weighting_beats_plain_counting():
    """3 confident SOBER vs 2 barely-NOT-SOBER frames: counting would say 60/40, weighting says sober."""
    eng = TemporalDecisionEngine()
    for i, p in enumerate([0.55, 0.05, 0.55, 0.05, 0.05]):
        d = eng.update(p_not_sober=p, timestamp=i / FPS)
    assert d.score < 0.35 and d.assessment == Assessment.SOBER


def test_low_model_confidence_is_uncertain():
    eng = TemporalDecisionEngine()
    for i in range(6):
        d = eng.update(p_not_sober=0.52 if i % 2 else 0.48, timestamp=i / FPS)
    assert d.assessment == Assessment.UNCERTAIN
    assert REASON_LOW_CONFIDENCE in d.reasons
    assert d.confidence is None


def test_label_and_probability_inputs_are_equivalent():
    a, b = TemporalDecisionEngine(), TemporalDecisionEngine()
    for i in range(5):
        da = a.update(label=N, confidence=0.8, timestamp=i)
        db = b.update(p_not_sober=0.8, timestamp=i)
    assert da.score == pytest.approx(db.score)


# ---------------------------------------------------------------- quality / missing frames
def test_poor_quality_frames_cannot_produce_confident_decision():
    _, d = run([N] * 15, quality=0.2)
    assert d[-1].assessment == Assessment.UNCERTAIN
    assert REASON_INSUFFICIENT_VALID in d[-1].reasons
    assert d[-1].valid_frames == 0


def test_missing_face_frames_lead_to_uncertain():
    _, d = run([S, None, None, S, None, None, S, None, None, S, None, None, S, None, None])
    assert d[-1].assessment == Assessment.UNCERTAIN
    assert REASON_UNSTABLE_FACE in d[-1].reasons


def test_malformed_probability_is_ignored():
    eng = TemporalDecisionEngine()
    obs = eng.add_prediction(p_not_sober=float("nan"), timestamp=0)
    assert not obs.valid
    assert not eng.add_prediction(p_not_sober=1.7, timestamp=0.1).valid


# ---------------------------------------------------------------- hysteresis / continuous monitoring
def test_warning_is_held_while_camera_is_covered():
    eng, _ = run([N] * 6)
    assert eng.assessment == Assessment.POTENTIALLY_NOT_SOBER
    _, d = run([None] * 30, engine=eng, start=1.0)                # no usable frames for 3 s
    assert d[-1].assessment == Assessment.POTENTIALLY_NOT_SOBER


def test_warning_released_only_after_sustained_sober_evidence_and_hold_time():
    cfg = TemporalConfig(min_hold_seconds=2.0)
    eng, _ = run([N] * 6, engine=TemporalDecisionEngine(cfg))
    _, d = run([S] * 6, engine=eng, start=0.6)                   # quick recovery: still held
    assert d[-1].assessment == Assessment.POTENTIALLY_NOT_SOBER
    _, d = run([S] * 20, engine=eng, start=1.2)                  # sustained + hold time passed
    assert d[-1].assessment == Assessment.SOBER


def test_single_invalid_frame_does_not_change_state():
    eng, _ = run([S] * 6)
    d1 = eng.update(p_not_sober=0.5, quality=0.1, timestamp=0.6)  # invalid frame: window still valid
    assert d1.assessment == Assessment.SOBER


def test_continuous_monitoring_reassesses_after_decision():
    """After an initial SOBER decision, later impairment evidence is detected without a restart."""
    eng, d = run([S] * 10)
    assert d[-1].assessment == Assessment.SOBER
    _, d = run([N] * 6, engine=eng, start=1.0)
    assert d[-1].assessment == Assessment.POTENTIALLY_NOT_SOBER
    st = states(d)
    first_change = next(i for i, x in enumerate(d) if x.changed)
    assert Assessment.SOBER not in st[first_change:]          # escalates; never flips back to SOBER


def test_window_is_short_and_old_frames_expire():
    eng = TemporalDecisionEngine(TemporalConfig(max_window_seconds=1.0))
    for i in range(10):
        eng.add_prediction(label=S, confidence=0.9, timestamp=i * 0.1)
    eng.add_prediction(label=S, confidence=0.9, timestamp=5.0)   # 4 s later
    assert eng.make_decision().total_frames == 1


def test_reset_starts_a_new_assessment():
    eng, _ = run([N] * 6)
    eng.reset()
    assert eng.assessment == Assessment.ASSESSING
    assert eng.calculate_temporal_score() is None and not eng.is_window_ready()


def test_decision_is_serialisable():
    _, d = run([S] * 5)
    data = d[-1].to_dict()
    assert data["assessment"] == "SOBER" and isinstance(data["reasons"], list)


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError):
        TemporalConfig(sober_threshold=0.7, not_sober_threshold=0.6)
    with pytest.raises(ValueError):
        TemporalConfig(window_size=3, min_valid_frames=5)
