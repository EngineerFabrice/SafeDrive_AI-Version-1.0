-- Phase 1 (v2 schema): accounts, cooperatives, memberships, role profiles, audit log.
-- Additive only (CREATE TABLE IF NOT EXISTS). Intended for the new safedrive_ai_v2
-- database; it does not read or change the legacy safedrive_ai database.
-- Times are stored in UTC. Assistance, location and notification tables come in later phases.

CREATE TABLE IF NOT EXISTS users (
    id              INT          NOT NULL AUTO_INCREMENT,
    username        VARCHAR(60)  NOT NULL,
    email           VARCHAR(255) NOT NULL,                 -- stored lower-case
    password_hash   VARCHAR(255) NOT NULL,                 -- bcrypt
    role            ENUM('driver','umusare','manager','admin') NOT NULL DEFAULT 'driver',
    phone           VARCHAR(30)  NULL,
    is_active       TINYINT(1)   NOT NULL DEFAULT 1,
    last_login_at   DATETIME(3)  NULL,
    created_at      DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    updated_at      DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
    PRIMARY KEY (id),
    UNIQUE KEY uq_users_email (email),
    KEY idx_users_role (role)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS cooperatives (
    id              INT          NOT NULL AUTO_INCREMENT,
    name            VARCHAR(120) NOT NULL,
    code            VARCHAR(20)  NOT NULL,                 -- short unique code, e.g. KGL-01
    district        VARCHAR(80)  NULL,
    status          ENUM('APPROVED','SUSPENDED') NOT NULL DEFAULT 'APPROVED',
    created_by      INT          NULL,
    created_at      DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    updated_at      DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
    PRIMARY KEY (id),
    UNIQUE KEY uq_cooperatives_name (name),
    UNIQUE KEY uq_cooperatives_code (code),
    CONSTRAINT fk_cooperatives_created_by FOREIGN KEY (created_by) REFERENCES users (id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Every driver, Umusare and manager belongs to exactly one cooperative (UNIQUE user_id).
-- Membership is for identity, verification and accountability; it is NOT a matching filter.
CREATE TABLE IF NOT EXISTS cooperative_memberships (
    id              INT          NOT NULL AUTO_INCREMENT,
    user_id         INT          NOT NULL,
    cooperative_id  INT          NOT NULL,
    member_role     ENUM('driver','umusare','manager') NOT NULL,
    status          ENUM('PENDING','APPROVED','REJECTED','REVOKED') NOT NULL DEFAULT 'PENDING',
    reviewed_by     INT          NULL,
    reviewed_at     DATETIME(3)  NULL,
    review_note     VARCHAR(255) NULL,
    created_at      DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    updated_at      DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
    PRIMARY KEY (id),
    UNIQUE KEY uq_memberships_user (user_id),
    KEY idx_memberships_coop_role (cooperative_id, member_role, status),
    CONSTRAINT fk_memberships_user FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
    CONSTRAINT fk_memberships_coop FOREIGN KEY (cooperative_id) REFERENCES cooperatives (id) ON DELETE RESTRICT,
    CONSTRAINT fk_memberships_reviewer FOREIGN KEY (reviewed_by) REFERENCES users (id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS driver_profiles (
    user_id         INT          NOT NULL,
    license_number  VARCHAR(50)  NULL,
    created_at      DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    updated_at      DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
    PRIMARY KEY (user_id),
    CONSTRAINT fk_driver_profiles_user FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- An Umusare is eligible for matching only when verification_status='VERIFIED'
-- AND availability='AVAILABLE' (enforced by the matching engine, a later phase).
CREATE TABLE IF NOT EXISTS umusare_profiles (
    user_id             INT          NOT NULL,
    license_number      VARCHAR(50)  NULL,
    verification_status ENUM('PENDING','VERIFIED','REJECTED','SUSPENDED') NOT NULL DEFAULT 'PENDING',
    verified_by         INT          NULL,
    verified_at         DATETIME(3)  NULL,
    availability        ENUM('OFFLINE','AVAILABLE','BUSY') NOT NULL DEFAULT 'OFFLINE',
    requests_received   INT          NOT NULL DEFAULT 0,   -- response-reliability inputs
    requests_accepted   INT          NOT NULL DEFAULT 0,
    created_at          DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    updated_at          DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
    PRIMARY KEY (user_id),
    KEY idx_umusare_eligible (verification_status, availability),
    CONSTRAINT fk_umusare_profiles_user FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
    CONSTRAINT fk_umusare_profiles_verifier FOREIGN KEY (verified_by) REFERENCES users (id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Append-only record of security- and safety-relevant actions.
CREATE TABLE IF NOT EXISTS audit_logs (
    id              BIGINT       NOT NULL AUTO_INCREMENT,
    occurred_at     DATETIME(3)  NOT NULL,
    actor_user_id   INT          NULL,                     -- NULL for anonymous/system events
    action          VARCHAR(60)  NOT NULL,                 -- e.g. LOGIN_SUCCEEDED, ROLE_CHANGED
    target_type     VARCHAR(40)  NULL,
    target_id       VARCHAR(64)  NULL,
    cooperative_id  INT          NULL,
    ip_address      VARCHAR(45)  NULL,
    details         VARCHAR(1000) NULL,                    -- short JSON; never passwords, images or locations
    PRIMARY KEY (id),
    KEY idx_audit_time (occurred_at),
    KEY idx_audit_actor (actor_user_id, occurred_at),
    KEY idx_audit_action (action, occurred_at),
    CONSTRAINT fk_audit_actor FOREIGN KEY (actor_user_id) REFERENCES users (id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
