# SafeDrive AI
### An AI-Based Real-Time Driver Sobriety Detection and Assistance System

Final-year Bachelor of Computer Engineering project — Fabrice NDAYISABA and Mugabo Kevia.

SafeDrive AI uses a driver-facing camera to estimate whether a driver may be affected by **alcohol**
and, when the assessment is *potentially not sober*, helps the driver request a verified **Umusare**
(support driver) from the cooperative network.

> **Scope and limits.** The project addresses alcohol-related impairment only. It outputs an AI
> assessment — **SOBER**, **UNCERTAIN** or **POTENTIALLY NOT SOBER** — and does **not** measure
> blood alcohol concentration, diagnose intoxication or replace a breathalyser. No accuracy figures
> are claimed until a model has been trained and evaluated on a verified dataset.

---

## Current status

| Area | Status |
|---|---|
| Camera capture (threaded, reconnect/backoff) | Implemented |
| Person detection (YOLOv8n) → driver ROI → face detection (YuNet) + landmarks | Implemented |
| 29 handcrafted facial features (baseline research features) | Implemented |
| Face quality gate + aligned face crop (shared by training and inference) | Implemented (`engine/preprocessing.py`) |
| Impairment-model interface | Implemented; feature-vector and face-image models |
| Temporal Decision Engine (short window, confidence-weighted, hysteresis) | Implemented; fed live by the classifier |
| MobileNetV3 alcohol classifier | **Prototype** trained on `AlcoholDetectionDataset` (one recording per class): integrated end-to-end, **not validated** on independent people (see `models/alcohol_mobilenetv3/README.md`) |
| Driver dashboard: live SOBER / UNCERTAIN / POTENTIALLY NOT SOBER, visual + audio alert | Implemented |
| Accounts, 4 roles, cooperatives, memberships, audit log | Implemented (`safedrive_ai_v2`) |
| Assistance workflow: request → nearby verified Umusare (any cooperative) → accept → live location → completion | Implemented (`website/assistance*.py`, `website/geo.py`) |
| Manager-side verification of drivers and Umusare (verify / reject / request info / suspend, audited) | Implemented (`website/cooperative_service.py`; the demo Umusare is pre-verified by the seed) |

## Architecture

```
Camera → Person detection → Driver ROI → Face detection → Face quality → Face crop → MobileNetV3 →
Temporal Decision Engine → SOBER / UNCERTAIN / POTENTIALLY_NOT_SOBER → Driver dashboard
POTENTIALLY_NOT_SOBER → visual + audio warning → REQUEST UMUSARE ASSISTANCE → nearby verified Umusare
  → ACCEPTED → live location (driver ↔ accepted Umusare only) → DRIVER_CONNECTED → COMPLETED
```

- `engine/` — real-time computer-vision engine, independent of Flask
  (`camera.py`, `detectors/`, `preprocessing.py`, `features/`, `impairment/`, `decision/`, `pipeline.py`).
- `training/` — dataset audit/preparation, MobileNetV3 transfer learning (frozen head, then fine-tuning), evaluation.
- `website/` — Flask app: accounts and roles (`routes.py`, `auth.py`), configuration (`config.py`),
  monitoring API (`monitoring.py`), history (`history.py`), audit log (`audit.py`), migrations.
- `tests/` — pytest suite (engine and web).
- `docs/` — dataset audit and model-integration notes.

### Roles and cooperatives
Roles: **driver**, **umusare**, **manager**, **admin**. Every driver and Umusare belongs to a
cooperative. An Umusare must be verified by their cooperative manager before receiving requests.
Cooperative membership is for identity and accountability — it is **not** a matching filter: a nearby,
verified, available Umusare from another cooperative can help any driver (same cooperative will only
be a ranking bonus).

## Setup

Requires Python 3.10 and MySQL 8.

```bash
pip install -r requirements.txt
cp .env.example .env                    # then set SAFEDRIVE_SECRET_KEY and database credentials
python -m website.migrate --create-database   # creates/migrates safedrive_ai_v2 (additive only)
python create_admin.py                  # first administrator (prompts for credentials)
python main.py                          # http://127.0.0.1:5000
```

The administrator creates the first cooperative; drivers and Umusare then register and choose it.

### Development / demo accounts

> ⚠ **DEVELOPMENT / DEMO CREDENTIALS ONLY.** These passwords are public (they are in this README
> and in `website/dev_seed.py`). Never create them on a production or internet-facing system; the
> seed command refuses to run when `SAFEDRIVE_ENV=production`.

```bash
python -m website.dev_seed      # creates missing demo accounts in safedrive_ai_v2; safe to run again
```

Login: http://127.0.0.1:5000/login

| Role | Email | Development password |
|---|---|---|
| Admin | admin@safedrive.ai | SafeDriveAdmin2026! |
| Manager | manager@safedrive.ai | SafeDriveManager2026! |
| Driver | driver@safedrive.ai | SafeDriveDriver2026! |
| Umusare | umusare@safedrive.ai | SafeDriveUmusare2026! |

The command is idempotent: existing accounts (matched by email) are never modified, and no
duplicates are created. Manager, Driver and Umusare belong to the approved **SafeDrive Demo
Cooperative** (`DEMO-01`); the demo Umusare is pre-verified by the demo admin (availability OFFLINE)
because the manager verification workflow is not built yet. Passwords are bcrypt-hashed with the
same validation as normal registration.

## Assistance workflow

States: `REQUESTED → MATCHING → ACCEPTED → DRIVER_CONNECTED → COMPLETED`, plus `CANCELLED` (driver, before
the Umusare has connected) and `NO_UMUSARE_AVAILABLE`. A request is created only when the driver presses
**REQUEST UMUSARE ASSISTANCE**; one active request per driver.

- **Matching** (server-side, `website/geo.py`): verified, active, available Umusare with an approved
  cooperative membership and a recently shared position, ranked by great-circle distance. Same cooperative
  is only a bonus of 0.5 km; any cooperative may assist. The radius widens (`ASSISTANCE_SEARCH_RADII_KM`) and
  unanswered offers expire; if nobody is found the driver sees `NO_UMUSARE_AVAILABLE` with manager contacts.
- **Privacy:** before acceptance an Umusare sees only a ~1 km area and a rounded distance. Exact positions
  are shared only between the driver and the accepted Umusare while the request is active; the exact pickup
  point and live positions are deleted when it ends. No location history; no coordinates in the audit log.
- **Journey, fare and payment:** the Umusare presses ARRIVED (journey starts) and COMPLETE JOURNEY. The server
  measures the distance from the Umusare's live updates (running total only, GPS jumps and inaccurate fixes ignored,
  never shorter than the straight line) and calculates `fare = base fee + km × price per km` (limits, whole RWF).
  Pricing is set by the admin (initially **RWF 500/km**, no base fee) and copied onto each completed journey.
  Payment: `PAYMENT_PENDING → PAYMENT_SENT` (driver, after paying externally, e.g. Mobile Money) `→ PAYMENT_COMPLETED`
  (Umusare confirms receipt); `PAYMENT_DISPUTED` if the Umusare reports a problem. SafeDrive does not move money.
- **Maps:** Leaflet with a configurable tile provider (`MAP_TILE_URL`, `MAP_TILE_ATTRIBUTION`, `MAP_TILE_SUBDOMAINS`,
  `MAP_TILE_MAX_ZOOM`, `MAP_TILE_API_KEY`); default CARTO Voyager (OpenStreetMap data, no key). The public
  tile.openstreetmap.org servers are not used (they block apps with 403). If tiles fail, a text fallback is shown. Admins manage drivers, Umusare (verification, availability, phone), assistance and pricing under /admin.
- **Google Maps is optional.** Without `GOOGLE_MAPS_API_KEY` the UI shows plain "open in / navigate with
  Google Maps" links; with it, embedded maps. Matching never depends on Google.

## Cooperatives, verification and internal chat

- **Manager ownership:** each cooperative has an assigned manager (`cooperatives.manager_user_id`); a manager
  belongs to exactly one cooperative and only sees that cooperative (other cooperatives' members answer 404).
  Admins create/edit/(de)activate cooperatives and assign or change managers under `/admin/cooperatives`.
- **Verification:** drivers and Umusare start `PENDING`; the cooperative manager (or an admin override) can
  VERIFY, REJECT (reason required), REQUEST MORE INFORMATION or SUSPEND. Verifier, time and cooperative are stored
  and every decision is audited. Unverified users can still sign in and see their manager's phone/email (from the
  database) with *Call*, *Email* and *Open Chat*. Unverified Umusare cannot go available; driver safety features
  (monitoring, requesting assistance) are never blocked. Verification never changes an AI sobriety result.
- **Chat (`/chat/`):** admins can message anyone; managers can message admins and their own cooperative; drivers and
  Umusare can message their manager, and once VERIFIED, other verified members of their own cooperative. Every call is
  authorized server-side; conversations use unguessable ids and contacts are signed per viewer. Unread counts and
  notifications use plain HTTP polling. Message text never goes to the audit log.
- **Location privacy is unchanged:** managers see operational status (ONLINE, AVAILABLE, ASSISTING, OFFLINE…) and
  ~1 km areas of active assistance only, never coordinates or history.

## Accounts, groups, vehicles and nearby drivers

- **Registration:** name, email, optional phone, role, cooperative, optional group, driver vehicle, and explicit
  acceptance of the [Terms](/terms) and [Privacy Policy](/privacy) (versioned; every acceptance is kept in
  `legal_acceptances`). Then a 6-digit **email OTP** (10 min, single use, 5 attempts, 60 s resend cooldown, 5 codes/hour,
  stored only as an HMAC). Email delivery uses `MAIL_*` settings (see `.env.example`); in development, without
  `MAIL_HOST`, emails are written to `instance/dev_outbox/` instead of being sent.
- **Separate states:** email verified, terms accepted, cooperative membership, manager verification, account active,
  AI safety status and operational status are stored and shown separately. The blue **✓ VERIFIED** badge appears only
  when all of a role's conditions hold (driver: email + membership + plate + manager approval + active) and lists its basis.
- **Groups:** Admin → Cooperative → Manager → Groups → Drivers/Umusare. Managers create and manage groups of their own
  cooperative only; the database itself prevents a member from joining another cooperative's group.
- **Vehicles:** drivers must add a plate number (normalised, e.g. `RAB 123 A`) before they can be verified; changing the
  plate of a verified driver sends them back for re-verification. The accepted Umusare sees the plate.
- **Nearby drivers (opt-in):** verified drivers can see other verified, opted-in drivers within 10 km as aggregated
  ~1 km grid cells (name, cooperative, group). Lookups use only the server's last accepted ~1 km cell for the driver
  (`POST /driver/presence` proposes an update: at most one per 15 s, implausible jumps rejected; `POST /driver/nearby`
  takes no coordinates and is rate-limited). Only the grid cell is stored (no history); exact positions are never returned.

## Tests

```bash
pytest -q
```

Database tests use a separate `safedrive_ai_v2_test` database (created automatically) and refuse to
run against any database whose name does not end in `_test`.

## Notes
- The old Keras scripts (`train_driver_model.py`, `Drunking_Detection_model.py`) and the bundled image
  folders are legacy and unused; see `docs/phase3_dataset_audit.md` for why they are not valid training data.
- YOLOv8n weights are AGPL-3.0 (Ultralytics); YuNet is MIT (OpenCV Zoo). See `models/README.md`.

## Credits
**SafeDrive AI — created/developed by Fabrice NDAYISABA**
**Email: [fabricendayisaba16@gmail.com](mailto:fabricendayisaba16@gmail.com)**
