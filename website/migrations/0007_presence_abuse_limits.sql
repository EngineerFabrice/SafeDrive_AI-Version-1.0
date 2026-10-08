-- Abuse limits for the nearby-driver presence (website/nearby_service.py). Additive only:
-- new defaulted columns on the existing driver_presence table; no data is changed or removed.
--   last_attempt_at      last location update received (accepted or rejected): minimum interval
--   lookup_window_start  start of the current nearby-lookup rate-limit window
--   lookup_count         nearby lookups made in that window
-- updated_at keeps its meaning: time of the last ACCEPTED (plausible) location update.
ALTER TABLE driver_presence
    ADD COLUMN last_attempt_at      DATETIME(3)  NULL,
    ADD COLUMN lookup_window_start  DATETIME(3)  NULL,
    ADD COLUMN lookup_count         INT          NOT NULL DEFAULT 0
