-- Cooperative manager ownership, member verification, internal chat and notifications.
-- Additive only: new columns (all NULL or with defaults), new tables, and a backfill that
-- copies existing approved manager memberships. Nothing is dropped, reset or deleted.

-- The manager responsible for a cooperative (its contact person). Managers still need an APPROVED
-- 'manager' membership in that cooperative for access; UNIQUE: one cooperative per assigned manager.
ALTER TABLE cooperatives
    ADD COLUMN manager_user_id INT NULL,
    ADD UNIQUE KEY uq_cooperatives_manager (manager_user_id),
    ADD CONSTRAINT fk_cooperatives_manager FOREIGN KEY (manager_user_id) REFERENCES users (id) ON DELETE SET NULL;

-- Existing data: the earliest approved manager member of each cooperative becomes its assigned manager.
UPDATE cooperatives c SET c.manager_user_id = (
    SELECT MIN(m.user_id) FROM cooperative_memberships m JOIN users u ON u.id = m.user_id
    WHERE m.cooperative_id = c.id AND m.member_role = 'manager' AND m.status = 'APPROVED' AND u.role = 'manager')
WHERE c.manager_user_id IS NULL;

-- Driver verification (same states as Umusare). Existing drivers start PENDING: no verification
-- has ever been recorded for them, so none is invented here.
ALTER TABLE driver_profiles
    ADD COLUMN verification_status      ENUM('PENDING','VERIFIED','REJECTED','SUSPENDED') NOT NULL DEFAULT 'PENDING',
    ADD COLUMN verified_by              INT          NULL,
    ADD COLUMN verified_at              DATETIME(3)  NULL,
    ADD COLUMN verified_cooperative_id  INT          NULL,
    ADD COLUMN reviewed_by              INT          NULL,
    ADD COLUMN reviewed_at              DATETIME(3)  NULL,
    ADD COLUMN verification_note        VARCHAR(255) NULL,
    ADD COLUMN info_requested_at        DATETIME(3)  NULL,
    ADD KEY idx_driver_verification (verification_status),
    ADD CONSTRAINT fk_driver_profiles_verifier FOREIGN KEY (verified_by) REFERENCES users (id) ON DELETE SET NULL,
    ADD CONSTRAINT fk_driver_profiles_reviewer FOREIGN KEY (reviewed_by) REFERENCES users (id) ON DELETE SET NULL,
    ADD CONSTRAINT fk_driver_profiles_vcoop FOREIGN KEY (verified_cooperative_id) REFERENCES cooperatives (id) ON DELETE SET NULL;

-- Review details for Umusare (verification_status / verified_by / verified_at already exist).
ALTER TABLE umusare_profiles
    ADD COLUMN verified_cooperative_id  INT          NULL,
    ADD COLUMN reviewed_by              INT          NULL,
    ADD COLUMN reviewed_at              DATETIME(3)  NULL,
    ADD COLUMN verification_note        VARCHAR(255) NULL,
    ADD COLUMN info_requested_at        DATETIME(3)  NULL,
    ADD CONSTRAINT fk_umusare_profiles_reviewer FOREIGN KEY (reviewed_by) REFERENCES users (id) ON DELETE SET NULL,
    ADD CONSTRAINT fk_umusare_profiles_vcoop FOREIGN KEY (verified_cooperative_id) REFERENCES cooperatives (id) ON DELETE SET NULL;

-- Internal chat. Conversations are addressed by an unguessable public_id, never the numeric id.
-- direct_key ('<lower user id>:<higher user id>') keeps one direct conversation per pair of users.
CREATE TABLE IF NOT EXISTS conversations (
    id                  INT          NOT NULL AUTO_INCREMENT,
    public_id           CHAR(22)     NOT NULL,
    conversation_type   ENUM('DIRECT','SUPPORT','VERIFICATION','ADMINISTRATIVE') NOT NULL DEFAULT 'DIRECT',
    direct_key          VARCHAR(40)  NULL,
    cooperative_id      INT          NULL,                    -- shared cooperative, NULL for admin conversations
    created_by          INT          NULL,
    created_at          DATETIME(3)  NOT NULL,
    updated_at          DATETIME(3)  NOT NULL,
    last_message_at     DATETIME(3)  NULL,
    PRIMARY KEY (id),
    UNIQUE KEY uq_conversations_public (public_id),
    UNIQUE KEY uq_conversations_direct (direct_key),
    KEY idx_conversations_last (last_message_at),
    CONSTRAINT fk_conversations_creator FOREIGN KEY (created_by) REFERENCES users (id) ON DELETE SET NULL,
    CONSTRAINT fk_conversations_coop FOREIGN KEY (cooperative_id) REFERENCES cooperatives (id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS conversation_participants (
    conversation_id     INT          NOT NULL,
    user_id             INT          NOT NULL,
    joined_at           DATETIME(3)  NOT NULL,
    last_read_at        DATETIME(3)  NULL,                    -- unread = messages from others after this
    PRIMARY KEY (conversation_id, user_id),
    KEY idx_participants_user (user_id),
    CONSTRAINT fk_participants_conversation FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE,
    CONSTRAINT fk_participants_user FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Message text only: no locations, attachments or other personal data are stored with a message.
CREATE TABLE IF NOT EXISTS messages (
    id                  BIGINT       NOT NULL AUTO_INCREMENT,
    conversation_id     INT          NOT NULL,
    sender_id           INT          NULL,
    body                VARCHAR(2000) NOT NULL,
    created_at          DATETIME(3)  NOT NULL,
    deleted_at          DATETIME(3)  NULL,
    PRIMARY KEY (id),
    KEY idx_messages_conversation (conversation_id, id),
    CONSTRAINT fk_messages_conversation FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE,
    CONSTRAINT fk_messages_sender FOREIGN KEY (sender_id) REFERENCES users (id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Lightweight in-app notifications (verification decisions, new verification requests).
-- Titles never contain message bodies or locations.
CREATE TABLE IF NOT EXISTS notifications (
    id                  BIGINT       NOT NULL AUTO_INCREMENT,
    user_id             INT          NOT NULL,
    kind                VARCHAR(40)  NOT NULL,
    title               VARCHAR(200) NOT NULL,
    link                VARCHAR(255) NULL,
    created_at          DATETIME(3)  NOT NULL,
    read_at             DATETIME(3)  NULL,
    PRIMARY KEY (id),
    KEY idx_notifications_user (user_id, read_at, created_at),
    CONSTRAINT fk_notifications_user FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
