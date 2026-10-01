# Phase 3 dataset audit: impairment model feasibility

Date: 2026-10-01 · Feature schema: v1 (29 features) · Reproduce: `python scripts/phase3_dataset_audit.py`

## Decision

**No impairment model was trained.** Neither dataset in the repository can
support a defensible alcohol-related impairment model. In both, the class
label is fully confounded with something other than impairment (one recording
per class, or one image source per class), and neither has ground truth for
alcohol intake. Any classifier trained on them, including the old CNNs,
would learn identity, session or source, and a validation score would only
measure that.

## Datasets found

| | AlcoholDetectionDataset | DrunkingDetectionDataset |
|---|---|---|
| Classes (files) | `alcoholic` 197, `non_alcoholic` 197 | `drunk` 240, `sober` 240 |
| Format | PNG, all 227×227 | JPG; drunk: 58 different sizes; sober: all 218×178 |
| Metadata / docs | none (no CSV/JSON, no provenance, no licence) | none, except Roboflow export names (`*.rf.<hash>.jpg`) |
| Subject IDs | none | none |
| Session/video IDs | none (filenames are frame numbers: A0523–A0719, a0007–a0255) | none |
| Used by | `train_driver_model.py` | `Drunking_Detection_model.py` |
| Old model artifacts | `driver_alcoholism_model.h5` not present | `Drunking_Detection_Model.h5` not present |

## What the labels actually represent

**AlcoholDetectionDataset.** Each class is a run of consecutive frames from
a **single video of a single person**:
- `alcoholic`: a man without glasses, in front of one background.
- `non_alcoholic`: a man wearing glasses, in front of a different background,
  with different lighting.

The frames are near-identical:
- Consecutive frames differ by a median dHash distance of 4 bits (alcoholic)
  and 7 bits (non_alcoholic), out of 64.
- All 197 `alcoholic` frames chain into a single near-duplicate cluster.
- The 197 `non_alcoholic` frames form 18 clusters.

The data does not show whether the two clips are the same person, nor what
the label is based on (measured BAC, self-report or staging). The effective
sample size is **1 recording per class**. The label equals "which video",
which is also "glasses or not" and "which background".

**DrunkingDetectionDataset.**
- `drunk`: about 60 web images (some groups were split by naming quirks, so 62
  filename groups), each exported 4 times by Roboflow (`…B`, `B1`, `B2`, `B3`:
  the original plus rotated copies, checked visually). They include
  police booking photos (one with a "Milwaukee County Sheriff's Office"
  watermark), paparazzi/party photos, people holding drinks, and faces whose
  red cheeks appear digitally added. Most likely labelled from context
  (arrest, party, appearance), not from measured intoxication.
- `sober`: 240 **CelebA** images (aligned 218×178 crops with 6-digit
  filenames): celebrities at public events.

The label equals "image source". **A rule that only checks whether the image
is 218×178 separates the classes with 100% accuracy** (480/480), without
looking at the face.

Neither dataset's labels represent alcohol-related impairment in a
measurable or verifiable sense.

## Quality and leakage findings

- **Exact duplicates:** 0 in both datasets.
- **Unreadable files:** 0.
- **Augmentation leakage:** `drunk` has 4 copies of every source image, so
  240 files are about 60 independent images. Any random image-level split
  puts copies of the same photo in train and test. The old
  `Drunking_Detection_model.py` uses Keras `validation_split=0.2`, which takes
  a contiguous slice of the sorted filenames per class. The copies are adjacent,
  so most stay together and only groups at the slice boundary leak. Its
  results are still invalid because of the source shortcut described above.
- **Session leakage:** the Alcohol dataset is 2 video clips. The old
  `train_driver_model.py` used a random `train_test_split` over frames, so
  its validation set was near-copies of its training frames, and any
  accuracy it reported says nothing about impairment.
- **Shortcuts:**
  - Image size, crop and alignment (Drunking).
  - Identity, glasses, background and lighting (Alcohol).
  - Photo genre: mugshot or party versus red carpet (Drunking).
  - Possible digital colour edits on `drunk` faces.
- **Grouped split:** impossible to do meaningfully.
  - Alcohol has 1 group per class, so leave-one-session-out trains on a single class.
  - Drunking can be grouped by source image, but the source and label
    confound remains, so a grouped split does not remove the shortcut.
- **Class balance:** 197/197 and 240/240 (balanced by count only).
- **Ethics:** the data contains identifiable real people (celebrities and
  booking photos) without documented consent or licence. Labelling identifiable
  individuals as "drunk" or "alcoholic" is itself a concern.

## Compatibility with the Phase 2 extractor

Run through the existing extractor (schema v1, unchanged). Images are tight
face crops, so if the person → ROI → face path finds no face, the face detector
is run on the whole image.

| Class | Face found | Features available | Gaze valid | Motion valid |
|---|---|---|---|---|
| alcoholic | 197/197 | 197 | 196 | 0 |
| non_alcoholic | 197/197 | 197 | 76 | 0 |
| drunk | 239/240 | 225 | 229 | 0 |
| sober | 240/240 | 233 | 229 | 0 |

- The extractor works technically on these images.
- **All 9 `facial_motion` features are always missing.** These are still
  images, or frames without timestamps, so 9 of 29 features cannot exist for
  this data.
- In `non_alcoholic`, gaze is valid for only 76/197 frames, because the
  glasses reflect light. That gap is itself a class-correlated artefact.
- The features do vary across samples (median per-feature std 0.36–0.72),
  but within each dataset that variation tracks recording or source rather
  than impairment.

## What data a defensible model would need

1. **Ground truth:** measured breath or blood alcohol (BrAC/BAC) per
   session, with time of measurement. Labels like "drunk" or "alcoholic"
   applied to photos are not enough.
2. **A within-subject design:** the same people recorded sober (baseline)
   and after controlled alcohol intake. This separates the effect of alcohol
   from identity. It needs enough participants for grouped (subject-level)
   evaluation; dozens at least, not 1–2.
3. **The same setup in every condition:** driver-facing camera, lighting,
   seat position and session timing kept constant, with the order
   counterbalanced so that time of day or fatigue does not stand in for
   alcohol.
4. **Video with timestamps,** so that motion, gaze stability and Phase 4
   temporal features can be computed. Single photos cannot provide them.
5. **Subject IDs, session IDs, consent and ethics approval;**
   documented provenance and licence.

Public sources to consider, each needing the same audit before use:
- Datasets of perceived intoxication from online videos, such as DIF (labels
  are perceived or contextual, not measured).
- Thermal-camera intoxication studies with known subjects (a different sensor
  from this RGB camera).
- A small in-house protocol, under ethics approval, that collects data
  matching items 1–5.

## Status of old code

`train_driver_model.py` and `Drunking_Detection_model.py` are left in place
(not deleted) but should not be used. Their data splits leak, as described
above, and their models are not in the repository. The live monitoring path
does not use them (deprecated in Phase 2).
