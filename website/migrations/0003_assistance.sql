-- Assistance workflow: driver requests, Umusare offers, latest-only locations.
-- Additive only: new tables; existing tables and data are untouched. Times are UTC.

-- One row per assistance request. Exact pickup coordinates are private: they are shown only to the
-- accepted Umusare and are cleared when the request ends (only the ~1 km approximate area is kept).
CREATE TABLE IF NOT EXISTS assistance_requests (
    id                      INT           NOT NULL AUTO_INCREMENT,
    driver_id               INT           NOT NULL,
    driver_cooperative_id   INT           NULL,
    status                  ENUM('REQUESTED','MATCHING','ACCEPTED','DRIVER_CONNECTED','COMPLETED',
                                 'CANCELLED','NO_UMUSARE_AVAILABLE') NOT NULL DEFAULT 'REQUESTED',
    trigger_source          VARCHAR(40)   NOT NULL,              -- SAFETY_ALERT | DRIVER_INITIATED
    pickup_lat              DECIMAL(9,6)  NULL,
    pickup_lon              DECIMAL(9,6)  NULL,
    pickup_accuracy_m       FLOAT         NULL,
    approx_lat              DECIMAL(6,2)  NOT NULL,
    approx_lon              DECIMAL(6,2)  NOT NULL,
    search_radius_km        FLOAT         NULL,
    matching_round          INT           NOT NULL DEFAULT 0,
    accepted_umusare_id     INT           NULL,
    cancelled_by            INT           NULL,
    created_at              DATETIME(3)   NOT NULL,
    matched_at              DATETIME(3)   NULL,
    accepted_at             DATETIME(3)   NULL,
    connected_at            DATETIME(3)   NULL,
    completed_at            DATETIME(3)   NULL,
    cancelled_at            DATETIME(3)   NULL,
    ended_at                DATETIME(3)   NULL,
    updated_at              DATETIME(3)   NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
    -- "One active request per driver / one active assistance per Umusare" is enforced by the service
    -- with row locks (users / umusare_profiles FOR UPDATE), keeping ON DELETE CASCADE/SET NULL usable.
    PRIMARY KEY (id),
    KEY idx_assistance_driver (driver_id, status),
    KEY idx_assistance_umusare (accepted_umusare_id, status),
    KEY idx_assistance_status (status),
    CONSTRAINT fk_assistance_driver FOREIGN KEY (driver_id) REFERENCES users (id) ON DELETE CASCADE,
    CONSTRAINT fk_assistance_driver_coop FOREIGN KEY (driver_cooperative_id) REFERENCES cooperatives (id) ON DELETE SET NULL,
    CONSTRAINT fk_assistance_umusare FOREIGN KEY (accepted_umusare_id) REFERENCES users (id) ON DELETE SET NULL,
    CONSTRAINT fk_assistance_cancelled_by FOREIGN KEY (cancelled_by) REFERENCES users (id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Umusare contacted for a request (one matching round offers the request to the best few candidates).
CREATE TABLE IF NOT EXISTS assistance_offers (
    id                  INT          NOT NULL AUTO_INCREMENT,
    request_id          INT          NOT NULL,
    umusare_id          INT          NOT NULL,
    matching_round      INT          NOT NULL,
    rank_in_round       INT          NOT NULL,
    distance_km         FLOAT        NOT NULL,
    same_cooperative    TINYINT(1)   NOT NULL DEFAULT 0,
    score_km            FLOAT        NOT NULL,
    status              ENUM('OFFERED','DECLINED','ACCEPTED','EXPIRED','WITHDRAWN') NOT NULL DEFAULT 'OFFERED',
    offered_at          DATETIME(3)  NOT NULL,
    responded_at        DATETIME(3)  NULL,
    PRIMARY KEY (id),
    UNIQUE KEY uq_offer_request_umusare (request_id, umusare_id),
    KEY idx_offer_umusare_status (umusare_id, status),
    CONSTRAINT fk_offer_request FOREIGN KEY (request_id) REFERENCES assistance_requests (id) ON DELETE CASCADE,
    CONSTRAINT fk_offer_umusare FOREIGN KEY (umusare_id) REFERENCES users (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Latest position only (no history). An Umusare row exists only while AVAILABLE or BUSY; a driver
-- row exists only while an accepted assistance is active. Rows are deleted when sharing stops.
CREATE TABLE IF NOT EXISTS user_locations (
    user_id             INT          NOT NULL,
    lat                 DECIMAL(9,6) NOT NULL,
    lon                 DECIMAL(9,6) NOT NULL,
    accuracy_m          FLOAT        NULL,
    updated_at          DATETIME(3)  NOT NULL,
    PRIMARY KEY (user_id),
    CONSTRAINT fk_user_locations_user FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
