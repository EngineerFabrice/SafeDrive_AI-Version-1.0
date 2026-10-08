-- Groups under cooperatives, driver vehicles, email verification (OTP), versioned Terms/Privacy
-- acceptance, and privacy-safe nearby-driver presence. Additive only: new tables and new NULL /
-- defaulted columns. Nothing is dropped, reset or deleted, and no verification is invented for
-- existing users (their email_verified_at and terms fields start NULL).

-- Groups belong to exactly one cooperative. (id, cooperative_id) is unique so memberships can
-- reference both: a member can never sit in a group of another cooperative (enforced by the database).
CREATE TABLE IF NOT EXISTS cooperative_groups (
    id              INT          NOT NULL AUTO_INCREMENT,
    cooperative_id  INT          NOT NULL,
    name            VARCHAR(80)  NOT NULL,
    status          ENUM('ACTIVE','INACTIVE') NOT NULL DEFAULT 'ACTIVE',
    created_by      INT          NULL,
    created_at      DATETIME(3)  NOT NULL,
    updated_at      DATETIME(3)  NOT NULL,
    PRIMARY KEY (id),
    UNIQUE KEY uq_group_name (cooperative_id, name),
    UNIQUE KEY uq_group_coop (id, cooperative_id),
    CONSTRAINT fk_groups_coop FOREIGN KEY (cooperative_id) REFERENCES cooperatives (id) ON DELETE RESTRICT,
    CONSTRAINT fk_groups_creator FOREIGN KEY (created_by) REFERENCES users (id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- A member (driver / Umusare) may belong to one group of their own cooperative.
ALTER TABLE cooperative_memberships
    ADD COLUMN group_id           INT          NULL,
    ADD COLUMN group_assigned_by  INT          NULL,
    ADD COLUMN group_assigned_at  DATETIME(3)  NULL,
    ADD KEY idx_memberships_group (group_id, cooperative_id),
    ADD CONSTRAINT fk_memberships_group FOREIGN KEY (group_id, cooperative_id)
        REFERENCES cooperative_groups (id, cooperative_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
    ADD CONSTRAINT fk_memberships_group_by FOREIGN KEY (group_assigned_by) REFERENCES users (id) ON DELETE SET NULL;

-- Driver vehicle. The plate is required before a driver can be verified (enforced by the service).
-- nearby_visibility: the driver's opt-in to appear (approximately) to nearby verified drivers.
ALTER TABLE driver_profiles
    ADD COLUMN vehicle_plate_number  VARCHAR(16)  NULL,
    ADD COLUMN vehicle_make          VARCHAR(40)  NULL,
    ADD COLUMN vehicle_model         VARCHAR(40)  NULL,
    ADD COLUMN vehicle_type          VARCHAR(20)  NULL,
    ADD COLUMN vehicle_updated_at    DATETIME(3)  NULL,
    ADD COLUMN nearby_visibility     TINYINT(1)   NOT NULL DEFAULT 0,
    ADD KEY idx_driver_plate (vehicle_plate_number);

-- Email verification and the CURRENT accepted Terms / Privacy versions (history: legal_acceptances).
ALTER TABLE users
    ADD COLUMN email_verified_at     DATETIME(3)  NULL,
    ADD COLUMN terms_version         VARCHAR(10)  NULL,
    ADD COLUMN terms_accepted_at     DATETIME(3)  NULL,
    ADD COLUMN privacy_version       VARCHAR(10)  NULL,
    ADD COLUMN privacy_accepted_at   DATETIME(3)  NULL;

-- Every acceptance ever given (append-only): accepting a new version never overwrites an old record.
CREATE TABLE IF NOT EXISTS legal_acceptances (
    id              BIGINT       NOT NULL AUTO_INCREMENT,
    user_id         INT          NOT NULL,
    document        ENUM('TERMS','PRIVACY') NOT NULL,
    version         VARCHAR(10)  NOT NULL,
    accepted_at     DATETIME(3)  NOT NULL,
    ip_address      VARCHAR(45)  NULL,
    PRIMARY KEY (id),
    KEY idx_legal_user (user_id, document, accepted_at),
    CONSTRAINT fk_legal_user FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- One-time email codes. Only an HMAC of the code is stored; codes are single-use, short-lived,
-- attempt-limited, and a new code invalidates the previous one.
CREATE TABLE IF NOT EXISTS email_otps (
    id              BIGINT       NOT NULL AUTO_INCREMENT,
    user_id         INT          NOT NULL,
    purpose         VARCHAR(20)  NOT NULL,
    code_hash       CHAR(64)     NOT NULL,
    sent_to         VARCHAR(255) NOT NULL,
    created_at      DATETIME(3)  NOT NULL,
    expires_at      DATETIME(3)  NOT NULL,
    attempts        INT          NOT NULL DEFAULT 0,
    consumed_at     DATETIME(3)  NULL,
    invalidated_at  DATETIME(3)  NULL,
    PRIMARY KEY (id),
    KEY idx_otp_user (user_id, purpose, created_at),
    CONSTRAINT fk_otp_user FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Nearby-driver presence: ONLY the ~1.1 km grid cell (2 decimals), latest only, no history.
-- Exact coordinates are never stored here; rows are deleted when the driver opts out.
CREATE TABLE IF NOT EXISTS driver_presence (
    user_id         INT          NOT NULL,
    approx_lat      DECIMAL(6,2) NOT NULL,
    approx_lon      DECIMAL(6,2) NOT NULL,
    updated_at      DATETIME(3)  NOT NULL,
    PRIMARY KEY (user_id),
    KEY idx_presence_time (updated_at),
    CONSTRAINT fk_presence_user FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
