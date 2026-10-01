# Feature vector (schema version 1)

`FaceFeatures.to_vector()` returns a `float32` array of length 29, in the
fixed order of `engine.features.FEATURE_NAMES` (shown below). The order
is part of the schema: any change to names or order must bump
`FEATURE_SCHEMA_VERSION` in `extractor.py`.

These are per-frame measurements for a later trained model. No single value,
and no single frame, indicates alcohol impairment.

## Missing values

- Every slot is always present; an unavailable value is `NaN` in the vector
  and `null` in `FaceFeatures.to_dict()` / `/monitoring/status`.
- If a group's `valid` flag is false, **all** of that group's slots are `NaN`,
  even when some partial numbers were computed.
- When the frame as a whole is `FEATURES_UNAVAILABLE` (no landmarks, or invalid head pose or
  appearance), the vector can still be produced and is mostly or entirely `NaN`.
- `facial_motion` is `NaN` on the first frame after a face appears and after
  any gap longer than 0.5 s.

## Temporal use

Each `FaceFeatures` carries `timestamp` (wall clock of the capture),
`frame_id` and `schema_version`, so a later stage can keep a sequence
`t1 -> vector, t2 -> vector, ...`. Rates in `facial_motion` are per second,
so they do not depend on the frame rate.

## Features

| # | Name | Meaning |
|---|------|---------|
| 0 | head_pose.pitch | degrees, + = face tilted up (solvePnP, generic 3D face, approximate intrinsics) |
| 1 | head_pose.yaw | degrees, + = face turned towards image right |
| 2 | head_pose.roll | degrees, + = face rotated clockwise in the image |
| 3 | head_pose.yaw_ratio | nose x offset from eye midpoint / interocular distance |
| 4 | head_pose.pitch_ratio | nose height between eye line (0) and mouth line (1) |
| 5 | head_pose.roll_2d | eye-line angle in the image, degrees |
| 6 | head_pose.reprojection_error | mean 3D-fit error / interocular distance (fit quality) |
| 7 | head_pose.confidence | landmark score x fit quality, 0..1 |
| 8 | gaze.horizontal | sclera balance, -1..1, + = iris towards image right (mean of usable eyes) |
| 9 | gaze.vertical | dark-centroid offset across the eye line / patch half-height (less reliable) |
| 10 | gaze.horizontal_left | horizontal value for the eye on the image left |
| 11 | gaze.horizontal_right | horizontal value for the eye on the image right |
| 12 | gaze.eye_disagreement | abs(left - right); large means an unreliable frame |
| 13 | gaze.eye_contrast | iris/sclera contrast, 0..1 (quality) |
| 14 | appearance.skin_a | mean CIELAB a* of cheek skin (+ = redder) |
| 15 | appearance.skin_red_ratio | mean R/(R+G+B) of cheek skin |
| 16 | appearance.eye_region_a_rel | eye-region a* minus cheek a* |
| 17 | appearance.brightness | mean L* of the face, 0..100 (quality) |
| 18 | appearance.contrast | std of L* across the face (quality) |
| 19 | appearance.sharpness | variance of Laplacian at 128 px width (quality) |
| 20 | facial_motion.face_speed | face-centre speed, face widths per second |
| 21 | facial_motion.face_dx | signed horizontal displacement since the previous frame, face widths |
| 22 | facial_motion.face_dy | signed vertical displacement since the previous frame, face widths |
| 23 | facial_motion.scale_change | log(width / previous width) per second |
| 24 | facial_motion.landmark_speed | mean landmark speed, interocular distances per second |
| 25 | facial_motion.landmark_deformation | landmark motion after removing translation, IOD per second |
| 26 | facial_motion.pitch_rate | degrees per second |
| 27 | facial_motion.yaw_rate | degrees per second |
| 28 | facial_motion.roll_rate | degrees per second (wrapped at +/-180) |

Colour features depend on lighting, white balance and skin tone. Use them
relative to the same driver's own baseline and together with the quality features.
