# Model card: AlcoholMobileNetV3 (prototype)

**Status: DEVELOPMENT_ONLY prototype.** It demonstrates the complete SafeDrive AI chain
(camera → face → classifier → temporal decision → warning). It is **not** a validated
alcohol detector, does not measure blood alcohol concentration and is not a medical or legal test.

| | |
|---|---|
| Architecture | torchvision `mobilenet_v3_small`, ImageNet-pretrained, last layer → 2 classes (1.52 M parameters) |
| Classes | `non_alcoholic` (0), `alcoholic` (1 = positive, "potentially not sober") |
| Input | 224×224 aligned face crop from `engine.preprocessing.align_face_crop`, RGB, ImageNet mean/std |
| Training data | `AlcoholDetectionDataset` (394 frames; labels undocumented, `label_type=dataset_label_unverified`) |
| Split | per-class contiguous blocks in recording order, 5-frame buffers: train 266 / val 38 / test 50 (40 buffer frames unused) |
| Stage 1 (frozen backbone) | head only (592,898 trainable params), AdamW 1e-3, 20 epochs |
| Stage 2 (fine-tuning) | last 4 feature blocks + head (1,329,386 params), AdamW 1e-4, early-stopped at epoch 12 |
| Selection | validation F1 and F1 under brightness/greyscale/blur/colour-cast perturbations: both stages scored 1.0 everywhere (tie) → the simpler frozen-backbone model was kept |
| Test (once) | accuracy 1.00, precision 1.00, recall/sensitivity 1.00, specificity 1.00, F1 1.00, ROC-AUC 1.00, CM [[25,0],[0,25]] |
| Latency | ~21 ms per crop on this CPU (inference only); full pipeline ~5 FPS, dominated by YOLOv8n |
| Files | `model.pt` (weights + metadata, git-ignored, rebuild below), `selection.json`, `test_report.json` |

## Why the perfect scores are not evidence of alcohol detection

From `data/processed/alcohol_v1/audit/AUDIT.md`:

- Each class is **one recording of one person**: no subject or session IDs exist, so no
  subject-independent split is possible. Test frames are later frames of the *same* recordings
  (nearest perceptual-hash distance to a training frame: 0 bits).
- The classes differ in glasses, lighting, colour cast and background. A logistic regression on
  **colour statistics alone** or **sharpness alone** reaches **100 %** test accuracy without
  looking at the face.

The model therefore separates two recordings. On a new driver its output is unpredictable,
which is why the temporal engine, face-quality gate, UNCERTAIN state and the prototype notice in
the UI matter. A defensible evaluation needs a dataset with many subjects recorded both sober and
after measured alcohol intake.

## Rebuild

```bash
python -m training.prepare_dataset              # audit + aligned crops + manifest
python -m training.train_mobilenet --stage head
python -m training.train_mobilenet --stage finetune
python -m training.train_mobilenet --stage select
python -m training.evaluate                     # test split, once
```
