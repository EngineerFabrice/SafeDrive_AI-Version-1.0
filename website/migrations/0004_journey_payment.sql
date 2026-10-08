-- Journey distance, fares and the two-step payment confirmation. Additive only.

-- Admin-configurable pricing. One row (id = 1). Initial value: 500 RWF per km, no base fee.
CREATE TABLE IF NOT EXISTS pricing_settings (
    id              TINYINT       NOT NULL,
    currency        CHAR(3)       NOT NULL DEFAULT 'RWF',
    price_per_km    DECIMAL(10,2) NOT NULL DEFAULT 500.00,
    base_fee        DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    minimum_fare    DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    maximum_fare    DECIMAL(10,2) NULL,                      -- NULL = no maximum
    updated_by      INT           NULL,
    updated_at      DATETIME(3)   NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
    PRIMARY KEY (id),
    CONSTRAINT fk_pricing_updated_by FOREIGN KEY (updated_by) REFERENCES users (id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

INSERT IGNORE INTO pricing_settings (id, currency, price_per_km, base_fee, minimum_fare, maximum_fare)
VALUES (1, 'RWF', 500.00, 0.00, 0.00, NULL);

-- Journey + fare + payment on the existing request row (no separate location history).
-- journey_last_* is only the most recent point used to add up distance; it and the exact journey
-- start are cleared at completion, keeping only ~1 km approximate start/end and the total distance.
ALTER TABLE assistance_requests
    ADD COLUMN journey_started_at        DATETIME(3)   NULL,
    ADD COLUMN journey_start_lat         DECIMAL(9,6)  NULL,
    ADD COLUMN journey_start_lon         DECIMAL(9,6)  NULL,
    ADD COLUMN journey_last_lat          DECIMAL(9,6)  NULL,
    ADD COLUMN journey_last_at           DATETIME(3)   NULL,
    ADD COLUMN journey_last_lon          DECIMAL(9,6)  NULL,
    ADD COLUMN journey_distance_km       DECIMAL(9,3)  NOT NULL DEFAULT 0,
    ADD COLUMN journey_ended_at          DATETIME(3)   NULL,
    ADD COLUMN journey_start_approx      VARCHAR(20)   NULL,
    ADD COLUMN journey_end_approx        VARCHAR(20)   NULL,
    ADD COLUMN final_distance_km         DECIMAL(9,2)  NULL,
    ADD COLUMN fare_currency             CHAR(3)       NULL,
    ADD COLUMN fare_price_per_km         DECIMAL(10,2) NULL,
    ADD COLUMN fare_base_fee             DECIMAL(10,2) NULL,
    ADD COLUMN fare_minimum              DECIMAL(10,2) NULL,
    ADD COLUMN fare_maximum              DECIMAL(10,2) NULL,
    ADD COLUMN final_fare                DECIMAL(10,2) NULL,
    ADD COLUMN payment_status            ENUM('ESTIMATED','PAYMENT_PENDING','PAYMENT_SENT','PAYMENT_COMPLETED',
                                              'PAYMENT_DISPUTED','PAYMENT_CANCELLED') NOT NULL DEFAULT 'ESTIMATED',
    ADD COLUMN payment_phone             VARCHAR(30)   NULL,      -- Umusare's registered phone at completion
    ADD COLUMN payment_pending_at        DATETIME(3)   NULL,
    ADD COLUMN payment_sent_at           DATETIME(3)   NULL,
    ADD COLUMN payment_completed_at      DATETIME(3)   NULL,
    ADD COLUMN payment_disputed_at       DATETIME(3)   NULL,
    ADD COLUMN payment_dispute_note      VARCHAR(255)  NULL,
    ADD COLUMN rating                    TINYINT       NULL,
    ADD COLUMN rating_comment            VARCHAR(500)  NULL,
    ADD COLUMN rated_at                  DATETIME(3)   NULL,
    ADD COLUMN problem_report            VARCHAR(500)  NULL,
    ADD COLUMN problem_reported_at       DATETIME(3)   NULL,
    ADD KEY idx_assistance_payment (payment_status);
