# SafeDrive AI — Mobile App (Flutter)

Android app for the SafeDrive AI platform: driver monitoring with the phone camera, sobriety-status
display, Umusare assistance requests, Umusare availability and request handling, live location
sharing during an accepted assistance, and cooperative-manager / administrator dashboards.

**Created by Fabrice NDAYISABA** — <fabricendayisaba16@gmail.com>

> The camera model is a research prototype trained on a limited dataset. Its results
> (`SOBER`, `UNCERTAIN`, `POTENTIALLY_NOT_SOBER`) are visual-pattern estimates. They do **not**
> measure blood alcohol concentration and never prove intoxication.

---

## How it fits the existing system

The app talks to the existing Flask backend through a dedicated JSON API, `/api/mobile/v1`
(`website/mobile_api.py`). That API calls the **same service functions** as the web app, so roles,
ownership checks, cooperative scoping, the assistance state machine, fares and privacy rules are
enforced in one place. The web app and its routes are unchanged.

| Concern | How it works |
|---|---|
| Sign-in | `POST /auth/login` → bearer token (32 random bytes). Only SHA-256(token) is stored (`mobile_api_tokens`, migration `0008`). Token expires after 30 days (`MOBILE_TOKEN_TTL_DAYS`), is revoked at sign-out, and stops working if the account is deactivated. Kept on the phone in Android secure storage. |
| CSRF | The mobile API accepts **only** the `Authorization: Bearer` header, never the browser session cookie, so it is CSRF-exempt. Web routes keep CSRF protection (tested). |
| Registration | Same validation, rate limit, Terms acceptance and duplicate-email protection as the web form, then the same email OTP. |
| Phone camera | The phone uploads one JPEG frame about every second to `POST /monitoring/phone/frame`. The server runs the **same** engine path as the vehicle camera (YOLOv8n person → driver ROI → YuNet face → face-quality gate → MobileNetV3 → temporal decision). Every driver gets their own temporal window. Frames are processed in memory and never stored. |
| Vehicle camera | The engine attached to the server (the web dashboard's camera) can be started, stopped and watched from the app. Only the driver who started it sees its results. |
| AI-triggered request | The app may *claim* `AI_TRIGGERED`. The server accepts it only if this driver's own monitoring session (phone or vehicle) currently reports `POTENTIALLY_NOT_SOBER`. |
| Location | Requested only when needed, while the app is in use. Sent when a driver requests assistance and when an Umusare goes available. Live sharing runs only while the assistance is `ACCEPTED`/`DRIVER_CONNECTED` (the server refuses it otherwise). No background tracking. |
| Secrets | None in the app. It only knows the backend address. Gmail/SMTP passwords, `.env` and the Flask secret stay on the server. |

### Features by role

* **Driver**: verification status and badge checklist; phone-camera monitoring with live assessment;
  vehicle-camera control; safety alert → request assistance; track matching; see and call the
  accepted Umusare (identity verified by the cooperative); live journey distance and estimated fare;
  cancel; mark payment sent; rate the Umusare; report a problem; fallback manager contacts when no
  Umusare is available.
* **Umusare**: go available/offline (shares/deletes position); incoming offers with only an
  approximate area until accepted; accept/decline; "I am with the driver" (starts the journey);
  complete the journey (the server calculates distance and fare); confirm payment or report a
  payment problem.
* **Cooperative manager**: own cooperative only. Stats, active assistance, members, and
  verify / reject / request info / suspend.
* **Administrator**: read-only overview. Management stays in the web console.
* **Everyone**: notifications, profile (phone number), Terms re-acceptance, email verification,
  About screen.

Not in the app (web only): chat, groups, nearby drivers, pricing, and user/cooperative administration.

---

## Prerequisites

* Flutter 3.38+ (Dart 3.10+): `flutter doctor`
* Android SDK (platform 36, build-tools) and JDK 17. Accept the licences once:
  `flutter doctor --android-licenses`
* The SafeDrive backend running (see the repository README), with MySQL and the migrations applied.

## 1. Start the backend

From the repository root (Windows PowerShell shown):

```powershell
.\.venv\Scripts\Activate.ps1
python -m website.migrate            # applies 0008_mobile_api_tokens.sql (additive)
python main.py                       # http://127.0.0.1:5000 — emulator only
```

For a **physical phone**, the server must listen on the network as well:

```powershell
$env:SAFEDRIVE_HOST = "0.0.0.0"; python main.py
```

Then allow inbound TCP port 5000 in Windows Defender Firewall (private network), and find the
computer's Wi-Fi IP with `ipconfig` (e.g. `192.168.1.20`). Use `SAFEDRIVE_HOST=0.0.0.0` only on a
trusted development network.

## 2. Configure the API address

| Where the app runs | Backend address |
|---|---|
| Android emulator | `http://10.0.2.2:5000` (the default; 10.0.2.2 is the host computer) |
| Physical phone, same Wi-Fi | `http://<computer-LAN-IP>:5000`, e.g. `http://192.168.1.20:5000` |
| Deployed server | `https://your-domain` |

`localhost` on a phone means the phone itself, so it never reaches your computer.

Set the address either at build time:

```bash
flutter run --dart-define=SAFEDRIVE_API_URL=http://192.168.1.20:5000
```

or at runtime: on the sign-in screen tap the **server icon** → enter the address → **Test connection** → **Save**
(stored on the device).

Debug and profile builds allow plain HTTP for local development
(`android/app/src/debug|profile/AndroidManifest.xml`). **Release builds allow HTTPS only.**

## 3. Run

```bash
cd mobile
flutter pub get
flutter devices                                   # emulator or USB phone (USB debugging on)
flutter run                                        # emulator, default http://10.0.2.2:5000
flutter run --dart-define=SAFEDRIVE_API_URL=http://192.168.1.20:5000   # physical phone
```

Accounts: register a driver or Umusare in the app (or on the web). Managers and administrators are
appointed on the web (`python create_admin.py` for the first administrator). For development
without SMTP, verification emails are written to `instance/dev_outbox/`.

## 4. Test and build

```bash
cd mobile
flutter analyze
flutter test                                       # unit + widget tests (fake HTTP backend)
flutter build apk --debug                          # build/app/outputs/flutter-apk/app-debug.apk
flutter build apk --release --dart-define=SAFEDRIVE_API_URL=https://your-domain
```

Install a built APK on a USB-connected phone: `adb install -r build/app/outputs/flutter-apk/app-debug.apk`.

The release build is signed with the debug key (Flutter default). Configure your own signing key
before distributing it.

Backend tests for the mobile API: `python -m pytest tests/web/test_mobile_api.py` (from the repository root).

## 5. Testing on a physical Android phone (debug APK)

1. **Phone**: Settings → About phone → tap *Build number* 7× → Developer options → enable **USB debugging**.
   Connect by USB and accept the "Allow USB debugging" prompt. `adb devices` must list it as `device`.
2. **Backend on the LAN** (repository root, PowerShell). Optionally log one line per phone frame:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   python -m website.migrate
   $env:SAFEDRIVE_HOST = "0.0.0.0"; $env:SAFEDRIVE_PHONE_DEBUG_LOG = "1"; python main.py
   ```
   The startup log must show `MySQL connection successful` and no model-loading error. Allow inbound TCP 5000
   for *private* networks in Windows Defender Firewall, then get the PC's Wi-Fi IPv4 with `ipconfig`.
3. **Check reachability from the phone first**: open `http://<PC-IP>:5000/api/mobile/v1/meta` in the phone's
   browser. You must see JSON. If not, it is a network/firewall problem, not an app problem. Guest or
   "client isolation" Wi-Fi networks block this; use a normal home network or the PC's mobile hotspot.
4. **Install**:
   ```powershell
   & "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe" install -r mobile\build\app\outputs\flutter-apk\app-debug.apk
   ```
   (or `flutter run --dart-define=SAFEDRIVE_API_URL=http://<PC-IP>:5000` from `mobile/` for hot reload).
5. **In the app**: server icon → `http://<PC-IP>:5000` → *Test connection* must say "Connected" → *Save* → sign in.
6. **Permissions**: *Open monitoring* → *Start phone monitoring* → allow **Camera** ("While using the app").
   Location is asked only when requesting assistance / going available. If a permission was denied
   permanently: Settings → Apps → SafeDrive AI → Permissions.
7. **Camera conditions**: phone fixed (stand or holder) at arm's length facing you, head and shoulders in
   view, face evenly lit from the front, no strong backlight. Each frame card shows *Face quality* (needs ≥ 0.50)
   and *Counted in assessment*. A first result needs at least 5 counted frames within about 10 s.
8. **Logs**:
   * app: `adb logcat -s flutter`. Debug builds print one `[SafeDrive] frame N: … quality=… counted=… assessment=…`
     line per frame.
   * crashes / permissions: `adb logcat *:E | findstr /i "safedrive camera AndroidRuntime"`
   * backend: the `python main.py` console, one `phone frame user=… quality=… counted=…` line per frame
     when `SAFEDRIVE_PHONE_DEBUG_LOG=1` (development only; never image data).

| Symptom | Meaning / fix |
|---|---|
| "Cannot reach the SafeDrive server" | wrong IP, server on 127.0.0.1 only, firewall, or isolated Wi-Fi (step 3) |
| "No driver detected" | head and shoulders not in view; move the phone back |
| quality rejected `too_dark` / `blurred` / `face_too_small` | more front light, hold still, move closer |
| face visible, "Counted: no", score < 0.50 | lighting/sharpness; the assessment stays ASSESSING/UNCERTAIN by design |
| stays ASSESSING | fewer than 5 counted frames in the window |

## 6. Software tests vs. model evaluation

These are different things. Passing tests means the **software works**. It says nothing about whether the
model detects alcohol.

| What | Command | What it shows |
|---|---|---|
| Mobile API + fake detectors | `python -m pytest tests/web/test_mobile_api.py` | auth, roles, privacy, the assistance flow, frame handling |
| Real pipeline (YOLO, YuNet, MobileNetV3) | `python -m pytest tests/engine/test_phone_pipeline_real.py` | real frames flow end to end; `counted` matches the temporal rule; EXIF rotation applied. Uses **training** images, so this is not evidence of accuracy |
| Model evaluation on independent people | `python scripts/evaluate_independent_faces.py --data <set>` | performance on consented images independent of training: refusals, exclusion reasons, frame-level metrics, session-level confusion matrix with 95 % Wilson intervals and abstentions, quality-gate comparison, confound warnings |

The evaluation script needs a set you collect yourself: written consent, pseudonymous `subject_id`s, the
same people recorded both sober and after measured alcohol intake (breath/blood test), and none of them the
person in `AlcoholDetectionDataset`. Its layout is in the script's docstring. It refuses unconsented rows and
training-identical / near-duplicate images, and its report is labelled *exploratory evaluation*. Even when
the minimum design criteria are met, the report says the results still need independent scientific review.
Reports go to `instance/evaluations/` (git-ignored).

## Project layout

```
mobile/lib/
  main.dart                 app, theme, signed-in / signed-out root
  config.dart               API URL (dart-define), credits, AI disclaimer
  api/api_client.dart       JSON + bearer token client, error mapping
  state/app_state.dart      session (secure token storage, server address, user)
  models.dart               user model, assessment label styles
  services/location_service.dart   permission-aware one-shot location
  widgets/common.dart       cards, status chips, assessment card, disclaimer
  screens/auth/             sign-in, registration, email OTP, server settings
  screens/driver/           dashboard, monitoring (phone + vehicle camera), assistance
  screens/umusare/          availability, offers, active assistance, payments
  screens/manager/          cooperative console and member review
  screens/admin/            read-only overview
  screens/                  home (role switch), profile, notifications, about
```

## Mobile API reference (`/api/mobile/v1`)

Public: `GET /meta`, `GET /registration-options`, `POST /auth/login`, `POST /auth/register`,
`POST /auth/verify-email`, `POST /auth/resend-code`.

Signed in: `POST /auth/logout`, `GET /me`, `POST /me/accept-terms`, `POST /me/phone`,
`GET /notifications`, `POST /notifications/read`.

Driver: `POST /assistance/requests`, `GET /assistance/requests/current`,
`POST /assistance/requests/<id>/cancel|payment-sent|rate|report`.

Umusare: `GET /assistance/umusare/status`, `POST /assistance/umusare/availability`,
`POST /assistance/requests/<id>/accept|decline|payment-received|payment-problem`.

Both participants: `POST /assistance/requests/<id>/connect|complete|location`. The server enforces
which side may do what (for example, only the Umusare completes a journey).

Monitoring (driver, admin): `GET /monitoring/model`, `POST /monitoring/phone/start|stop`,
`GET /monitoring/phone/status`, `POST /monitoring/phone/frame` (raw `image/jpeg` body or multipart
field `frame`, ≤ 900 KB, ≤ 4 frames/s), `GET /monitoring/vehicle/status`,
`POST /monitoring/vehicle/start|stop`, `GET /monitoring/history`.

Manager: `GET /manager/console`, `POST /manager/members/<id>/review` (`VERIFY|REJECT|REQUEST_INFO|SUSPEND`).
Admin: `GET /admin/overview`.

## Known limitations

* Phone frames arrive at about 1 FPS (upload + CPU inference ≈ 120 ms on the development PC), so the
  phone temporal window spans up to 10 s instead of the vehicle camera's 3 s. Frame count,
  thresholds and hysteresis are unchanged.
* The face-quality gate rejects blurry, dark or small faces. Those frames are counted but not
  scored, so the result stays `ASSESSING`/`UNCERTAIN` rather than guessing.
* Two quality thresholds exist: a frame is *predicted* when the per-frame gate passes (`ok`), but the
  temporal engine only *counts* it when its quality score is ≥ 0.50. Training accepted every `ok` frame, so
  ~31 % of the "alcoholic" training recording (darker and blurrier: median brightness 87 vs 106, Laplacian
  sharpness 85 vs 263) would not be counted live. The quality gate is therefore correlated with the label.
  This is a property of the dataset. Thresholds were deliberately not lowered.
  Measured from `data/processed/alcohol_v1/manifest.csv`: 61/197 alcoholic frames are `ok` but score < 0.50
  (train 44/133, val 4/19, test 3/25, buffer 10/20) against 0/197 non-alcoholic frames. Their limiting
  terms are sharpness (median term 0.61) and lighting (0.76). Aligning the gates either way changes
  model behaviour and needs an independent evaluation set first. The evaluation report's
  `quality_gate_comparison` scores the same frames under both gates for that decision.
* Phone monitoring sessions live in server memory (they end after 2 minutes without frames or when
  the server restarts) and are not written to the monitoring history.
* Location is shared only while the relevant screen is open. There is no background service.
* The model card (`models/alcohol_mobilenetv3/README.md`) explains why its test scores are not
  evidence of alcohol detection on new drivers.
