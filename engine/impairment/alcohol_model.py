"""MobileNetV3 face-image classifier for alcohol-related visual patterns (provider ``alcohol_mobilenetv3``).

Architecture: torchvision MobileNetV3 (ImageNet-pretrained during training) with
its last linear layer replaced by a 2-class layer. The artefact written by
``training/train_mobilenet.py`` holds the weights plus metadata (classes,
input size, normalisation, dataset, metrics and limitations); this module
rebuilds the architecture without downloading anything and loads the weights.

Input: an aligned BGR face crop from ``engine.preprocessing.align_face_crop``.
Output: probabilities for ``non_alcoholic`` / ``alcoholic``. A single
frame's output is never a decision; the temporal decision engine aggregates
frames into SOBER / UNCERTAIN / POTENTIALLY_NOT_SOBER.
"""

import os
from typing import Dict, Optional, Tuple

import numpy as np

from ..features import FEATURE_SCHEMA_VERSION
from .interface import ImpairmentModel, ModelInfo

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL_PATH = os.path.join(_REPO_ROOT, "models", "alcohol_mobilenetv3", "model.pt")
ARCHITECTURES = ("mobilenet_v3_small", "mobilenet_v3_large")


def build_mobilenet_v3(arch: str = "mobilenet_v3_small", num_classes: int = 2, pretrained: bool = False):
    """MobileNetV3 with a ``num_classes`` output layer. ``pretrained`` downloads ImageNet weights."""
    import torch.nn as nn
    from torchvision import models

    if arch not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {arch!r}")
    weights = None
    if pretrained:
        weights = (models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if arch == "mobilenet_v3_small"
                   else models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    net = getattr(models, arch)(weights=weights)
    last = net.classifier[-1]
    net.classifier[-1] = nn.Linear(last.in_features, num_classes)
    return net


class AlcoholImageModel(ImpairmentModel):
    input_type = "face_image"

    def __init__(self, path: Optional[str] = None, provider: str = "alcohol_mobilenetv3"):
        import torch

        self.path = path or os.environ.get("ALCOHOL_MODEL_PATH") or DEFAULT_MODEL_PATH
        if not os.path.isabs(self.path):
            self.path = os.path.join(_REPO_ROOT, self.path)        # relative to the repository, not the cwd
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"model artefact not found: {self.path} (run training/train_mobilenet.py)")
        artefact = torch.load(self.path, map_location="cpu", weights_only=False)
        meta = artefact["metadata"]
        self.metadata = meta
        self.classes: Tuple[str, ...] = tuple(meta["classes"])
        self.positive_class = meta["positive_class"]
        self.input_size = int(meta["input_size"])
        self._mean = np.asarray(meta["mean"], np.float32).reshape(1, 1, 3)
        self._std = np.asarray(meta["std"], np.float32).reshape(1, 1, 3)

        # No torch.set_num_threads() here: it is process-wide and would also slow the YOLO detector.
        self._torch = torch
        self._net = build_mobilenet_v3(meta["arch"], len(self.classes), pretrained=False)
        self._net.load_state_dict(artefact["state_dict"])
        self._net.eval()

        self._info = ModelInfo(
            name="AlcoholMobileNetV3", version=str(meta.get("version", "1")), provider=provider,
            schema_version=FEATURE_SCHEMA_VERSION, classes=self.classes, is_mock=False,
            development_only=bool(meta.get("development_only", True)), input_type="face_image",
            positive_class=self.positive_class,
            description=meta.get("limitations", "Prototype model; not validated on independent subjects."))

    @property
    def info(self) -> ModelInfo:
        return self._info

    @property
    def result_note(self) -> str:
        return "Prototype model output; not a measurement of blood alcohol." if self._info.development_only else ""

    def preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        """BGR uint8 crop -> normalised CHW float32 (same as training: RGB, /255, ImageNet mean/std)."""
        import cv2
        if face_bgr.shape[:2] != (self.input_size, self.input_size):
            face_bgr = cv2.resize(face_bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        rgb = face_bgr[..., ::-1].astype(np.float32) / 255.0
        return ((rgb - self._mean) / self._std).transpose(2, 0, 1).copy()

    def _predict(self, face_bgr: np.ndarray) -> Tuple[str, Optional[Dict[str, float]]]:
        torch = self._torch
        with torch.inference_mode():
            logits = self._net(torch.from_numpy(self.preprocess(face_bgr))[None])
            probs = torch.softmax(logits, dim=1)[0].numpy()
        return self.classes[int(np.argmax(probs))], {c: float(p) for c, p in zip(self.classes, probs)}
