"""Alcohol image-model provider: artefact loading, preprocessing parity, interface validation.

Uses a randomly initialised MobileNetV3 saved to a temp file, so no download and
no trained artefact are needed.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from engine.impairment import ImpairmentInput, ImpairmentStatus, create_impairment_model  # noqa: E402
from engine.impairment.alcohol_model import AlcoholImageModel, build_mobilenet_v3          # noqa: E402

META = {"version": "test", "arch": "mobilenet_v3_small", "classes": ["non_alcoholic", "alcoholic"],
        "positive_class": "alcoholic", "input_size": 224, "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225], "development_only": True, "limitations": "test artefact"}


@pytest.fixture(scope="module")
def artefact(tmp_path_factory):
    torch.manual_seed(0)
    net = build_mobilenet_v3("mobilenet_v3_small", 2, pretrained=False)
    path = tmp_path_factory.mktemp("model") / "model.pt"
    torch.save({"state_dict": net.state_dict(), "metadata": META}, path)
    return str(path)


@pytest.fixture(scope="module")
def model(artefact):
    return AlcoholImageModel(artefact)


def crop(value=120):
    return np.full((224, 224, 3), value, np.uint8)


def test_model_info_declares_image_input_and_positive_class(model):
    info = model.info
    assert info.input_type == "face_image" and info.positive_class == "alcoholic"
    assert info.classes == ("non_alcoholic", "alcoholic")
    assert info.development_only and not info.is_mock


def test_prediction_returns_probabilities(model):
    r = model.predict(ImpairmentInput.from_face(crop(), frame_id=3))
    assert r.valid and r.frame_id == 3 and r.prediction in ("non_alcoholic", "alcoholic")
    assert sum(r.probabilities.values()) == pytest.approx(1.0, abs=1e-5)
    assert "not a measurement of blood alcohol" in r.note


def test_preprocessing_matches_training_normalisation(model):
    """BGR crop -> RGB / 255 -> ImageNet mean/std, CHW (same as torchvision ToTensor + Normalize)."""
    img = np.zeros((224, 224, 3), np.uint8)
    img[..., 2] = 255                                    # pure red in BGR
    x = model.preprocess(img)
    assert x.shape == (3, 224, 224)
    assert x[0, 0, 0] == pytest.approx((1.0 - 0.485) / 0.229, rel=1e-5)   # R channel first
    assert x[2, 0, 0] == pytest.approx((0.0 - 0.406) / 0.225, rel=1e-5)


def test_other_crop_sizes_are_resized(model):
    assert model.predict(ImpairmentInput.from_face(np.full((160, 160, 3), 90, np.uint8))).valid


@pytest.mark.parametrize("bad", [None, np.zeros((224, 224), np.uint8), np.zeros((224, 224, 3), np.float32),
                                 np.zeros((8, 8, 3), np.uint8)])
def test_invalid_face_images_rejected_before_the_network(model, bad):
    r = model.predict(ImpairmentInput.from_face(bad))
    assert r.status == ImpairmentStatus.INVALID_INPUT and r.prediction is None


def test_registry_loads_artefact_and_reports_missing_file(artefact, monkeypatch):
    monkeypatch.setenv("ALCOHOL_MODEL_PATH", artefact)
    loaded = create_impairment_model("alcohol_mobilenetv3")
    assert loaded.available and loaded.info.positive_class == "alcoholic"
    monkeypatch.setenv("ALCOHOL_MODEL_PATH", "does/not/exist.pt")
    missing = create_impairment_model("alcohol_mobilenetv3")
    assert not missing.available
    r = missing.predict(ImpairmentInput.from_face(crop()))
    assert r.status == ImpairmentStatus.MODEL_UNAVAILABLE and "not found" in r.error


def test_mock_model_never_declares_a_positive_class():
    assert create_impairment_model("mock").info.positive_class is None
