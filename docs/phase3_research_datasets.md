# Phase 3: research dataset investigation

Date: 2026-10-01. Sources are the official pages and papers listed in each
section. No data was downloaded and no model was trained. Statements marked
*not verified* could not be checked from the official source.

The Phase 2 schema (v1, 29 per-frame features) needs **continuous,
timestamped, driver-facing RGB face video**, with per-subject and per-session
labels backed by measured BAC/BrAC.

---

## 1. Keshtkaran et al., WACV 2024 (Edith Cowan University / MiX Telematics)

| Item | Finding |
|---|---|
| Paper | "Estimating Blood Alcohol Level Through Facial Features for Driver Impairment Assessment", WACV 2024, pp. 4539–4548 (CVF open access); PhD thesis at ro.ecu.edu.au/theses/2893 (2025) |
| Official dataset URL | **None.** No repository or download page exists. |
| Access | Paper §7: *"The data used in this paper will be made available on request, provided the use is for non-profit and research. A transfer agreement will be required and data may not be available before 2025."* Corresponding contact printed in the paper: e.keshtkaran@ecu.edu.au |
| Licence / terms | No public licence; terms are set by the data transfer agreement (non-profit research only). Redistribution: *not stated; assume not allowed.* |
| Subjects | 60 (aged 19–76, mean 43 ± 14.6); Edith Cowan University ethics approval; informed consent |
| Conditions | Simulated urban driving (10-minute drives). Sober drive, then two drives after controlled alcohol intake |
| Ground truth | Breathalyser readings (AlcoQuant 6020 plus a simulator-integrated Autowatch 720). Labels: sober 0.000; "low" 0.051–0.069 (mean 0.058); "severe" 0.081–0.165 (mean 0.104) g/100 ml |
| IDs | Driver ID via RFID card, trip timestamps, BAC record per trip, so subject and session structure exist |
| Video | Main RGB face camera (Hikvision, 2–5 MP, 30 fps); ZED 2 3D (1080p, 30 fps); IR DSM camera (960p, 30 fps); rear-view RGB (720p, 15 fps); screen recordings |
| Phase 2 compatibility | **High (on paper).** Continuous 30 fps RGB face video supports head pose, gaze, appearance (constant lab lighting) and facial motion with real timestamps. The authors used OpenFace head pose, gaze and landmark features, which is close to our feature groups. *Not verified on actual files.* |
| Leakage | Within-subject design (each subject has all 3 states), so grouped subject-level splits are possible and required. The paper's main evaluation splits by subject; its "robustness check" uses StratifiedKFold, which may not be subject-grouped. **Confound:** the order is always sober → low → severe, so BAC is confounded with time-on-task, fatigue and simulator habituation; there is no placebo or counterbalancing. |
| Suitability | **The best documented fit for Phase 3, if access is granted.** It is the only candidate with measured BAC, multiple levels, within-subject sessions and RGB face video. |

## 2. Toyota Research Institute: Impaired Driving Dataset (IDD)

| Item | Finding |
|---|---|
| Paper | Gideon et al., "A Simulator Dataset to Support the Study of Impaired Driving", arXiv:2507.02867 |
| Official dataset URL | https://toyotaresearchinstitute.github.io/IDD/. **The page could not be loaded from this network** (connection reset, both direct and via fetch). No public `ToyotaResearchInstitute/IDD` GitHub repository was found via the GitHub API. Search snippets say "available for download using scripts in the linked repo": *not verified*. |
| Licence / terms | *Not verified* (the official page was unreachable) |
| Subjects | 52: v1 n=20 (alcohol + cognitive distraction; full-time TRI employees), v2 n=32 (distraction only, no alcohol). WCG IRB #20241945 |
| Conditions | 23.7 h of CARLA simulator urban driving; alcohol (target BAC 0.10%), n-back and sentence cognitive-distraction tasks, 8 scripted road hazards |
| Ground truth | Breathalyser: BAC 0 confirmed before drinking; BAC logged every 10 min; driving resumed once BAC was above 0.1 and falling |
| Driver-facing data | Abstract: *"driver-facing data (gaze, audio, surveys)"*. Gaze is a **Tobii Spark Pro eye tracker at 60 Hz** (gaze vectors per eye, on-screen gaze point, pupil diameter). **No face video is described as released**; the paper stresses PII is kept on HIPAA-compliant machines. |
| Phase 2 compatibility | **None.** There are no face images, so the Phase 2 extractor cannot run. Tobii gaze vectors are a different sensor and do not match the schema's gaze features. |
| Leakage | Alcohol is only in v1 (20 people, within-subject sober/intoxicated) |
| Suitability | **Not suitable** for the Phase 2 RGB feature model. Potentially useful only for a separate gaze-signal study. |

### Related TRI dataset: "Beyond Breathalysers" (IEEE IV 2025)

- **Source:** github.com/ToyotaResearchInstitute/IV25-beyond-breathalysers.
- **Licence:** CC BY-NC 4.0 (repository LICENSE).
- **Data:** under 700 MB. 50 subjects in the paper (20 alcohol-impaired, 30
  control); the README says 51. Tobii Pro Spark gaze and pupil at 60 Hz, BAC by
  Alco-Sensor FST breathalyser. Short sobriety-test tasks, not driving video.
- **Access:** register an email through the Google Form linked in the README,
  or email the address listed there.
- **Fit:** the paper itself says "other modalities of data such as video
  would be beneficial", so this is **also not face video** and **not
  compatible** with Phase 2.

---

## Outcome

- **Usable for the frozen Phase 2 schema:** only Keshtkaran et al. (WACV 2024),
  and only after a data request and transfer agreement.
- **TRI IDD and IV25:** measured BAC, but eye-tracker signals instead of face
  video. Incompatible without a different, eye-tracker-based feature path,
  which is out of the current scope.
- **What the user must do manually:** request the WACV dataset from the
  authors. Phase 3 cannot proceed technically until the data is received and
  audited (format, frame rate, IDs, labels, extraction success rate).
