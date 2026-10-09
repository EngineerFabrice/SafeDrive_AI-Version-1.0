-- Bearer tokens for the SafeDrive AI mobile app (website/mobile_auth.py). Additive only.
-- Only SHA-256(token) is stored; the token itself is shown to the app once, at sign-in.
-- A token stops working when it expires, is revoked (sign-out), or its user is deactivated or deleted.
CREATE TABLE IF NOT EXISTS mobile_api_tokens (
    id            BIGINT       NOT NULL AUTO_INCREMENT,
    user_id       INT          NOT NULL,
    token_hash    CHAR(64)     NOT NULL,
    device_name   VARCHAR(80)  NULL,
    created_at    DATETIME(3)  NOT NULL,
    last_used_at  DATETIME(3)  NULL,
    expires_at    DATETIME(3)  NOT NULL,
    revoked_at    DATETIME(3)  NULL,
    PRIMARY KEY (id),
    UNIQUE KEY uq_mobile_token_hash (token_hash),
    KEY idx_mobile_token_user (user_id, revoked_at),
    CONSTRAINT fk_mobile_token_user FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
