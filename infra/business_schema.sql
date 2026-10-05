CREATE TABLE IF NOT EXISTS `agentrun` (
    `id` VARCHAR(36) NOT NULL PRIMARY KEY,
    `created_at` DATETIME(6) NOT NULL,
    `owner_id` VARCHAR(36) NOT NULL,
    `kb_id` VARCHAR(36) NOT NULL,
    `conversation_id` VARCHAR(36) NOT NULL,
    `query` LONGTEXT NOT NULL,
    `status` VARCHAR(20) NOT NULL,
    `answer` LONGTEXT NOT NULL,
    `citations` JSON NOT NULL,
    `grade_retries` INT NOT NULL,
    `check_retries` INT NOT NULL,
    `config_snapshot` JSON NOT NULL,
    `error` LONGTEXT NOT NULL,
    `elapsed_ms` INT NOT NULL,
    `request_snapshot` JSON NOT NULL,
    `lease_owner` VARCHAR(36) NOT NULL,
    `lease_until` DATETIME(6),
    `event_sequence` INT NOT NULL,
    KEY `idx_agentrun_owner_i_56c31a` (`owner_id`)
) CHARACTER SET utf8mb4;
CREATE TABLE IF NOT EXISTS `agentstep` (
    `id` VARCHAR(36) NOT NULL PRIMARY KEY,
    `created_at` DATETIME(6) NOT NULL,
    `run_id` VARCHAR(36) NOT NULL,
    `node` VARCHAR(30) NOT NULL,
    `ordinal` INT NOT NULL,
    `status` VARCHAR(20) NOT NULL,
    `input_summary` JSON NOT NULL,
    `output_summary` JSON NOT NULL,
    `elapsed_ms` INT NOT NULL,
    KEY `idx_agentstep_run_id_7756bf` (`run_id`)
) CHARACTER SET utf8mb4;
CREATE TABLE IF NOT EXISTS `application` (
    `id` VARCHAR(36) NOT NULL PRIMARY KEY,
    `created_at` DATETIME(6) NOT NULL,
    `owner_id` VARCHAR(36) NOT NULL,
    `kb_id` VARCHAR(36) NOT NULL,
    `name` VARCHAR(120) NOT NULL,
    `description` LONGTEXT NOT NULL,
    `strategy` VARCHAR(20) NOT NULL,
    `mode` VARCHAR(12) NOT NULL,
    `welcome` LONGTEXT NOT NULL,
    `fallback` LONGTEXT NOT NULL,
    `prompt` LONGTEXT NOT NULL,
    `enabled` BOOL NOT NULL,
    KEY `idx_application_owner_i_1e620c` (`owner_id`),
    KEY `idx_application_kb_id_1cf63c` (`kb_id`)
) CHARACTER SET utf8mb4;
CREATE TABLE IF NOT EXISTS `chatmessage` (
    `id` VARCHAR(36) NOT NULL PRIMARY KEY,
    `created_at` DATETIME(6) NOT NULL,
    `owner_id` VARCHAR(36) NOT NULL,
    `conversation_id` VARCHAR(36) NOT NULL,
    `run_id` VARCHAR(36) NOT NULL,
    `role` VARCHAR(12) NOT NULL,
    `content` LONGTEXT NOT NULL,
    `citations` JSON NOT NULL,
    `accepted` BOOL NOT NULL,
    KEY `idx_chatmessage_owner_i_ad1f14` (`owner_id`),
    KEY `idx_chatmessage_convers_2520e8` (`conversation_id`)
) CHARACTER SET utf8mb4;
CREATE TABLE IF NOT EXISTS `conversation` (
    `id` VARCHAR(36) NOT NULL PRIMARY KEY,
    `created_at` DATETIME(6) NOT NULL,
    `owner_id` VARCHAR(36) NOT NULL,
    `kb_id` VARCHAR(36) NOT NULL,
    `title` VARCHAR(120) NOT NULL,
    `application_id` VARCHAR(36) NOT NULL,
    `active_run_id` VARCHAR(36) NOT NULL,
    `active_run_until` DATETIME(6),
    KEY `idx_conversatio_owner_i_fd1fa4` (`owner_id`),
    KEY `idx_conversatio_applica_164044` (`application_id`)
) CHARACTER SET utf8mb4;
CREATE TABLE IF NOT EXISTS `document` (
    `id` VARCHAR(36) NOT NULL PRIMARY KEY,
    `created_at` DATETIME(6) NOT NULL,
    `owner_id` VARCHAR(36) NOT NULL,
    `kb_id` VARCHAR(36) NOT NULL,
    `filename` VARCHAR(255) NOT NULL,
    `file_type` VARCHAR(10) NOT NULL,
    `storage_path` LONGTEXT NOT NULL,
    `sha256` VARCHAR(64) NOT NULL,
    `status` VARCHAR(20) NOT NULL,
    `error` LONGTEXT NOT NULL,
    `tags` JSON NOT NULL,
    `metadata` JSON NOT NULL,
    `embedding_fingerprint` VARCHAR(160) NOT NULL,
    `child_count` INT NOT NULL,
    `parent_count` INT NOT NULL,
    `index_revision` INT NOT NULL,
    `updated_at` DATETIME(6) NOT NULL,
    UNIQUE KEY `uid_document_kb_id_615aa9` (`kb_id`, `sha256`),
    KEY `idx_document_owner_i_b43b8c` (`owner_id`),
    KEY `idx_document_kb_id_555982` (`kb_id`),
    KEY `idx_document_status_ff7f1e` (`status`)
) CHARACTER SET utf8mb4;
CREATE TABLE IF NOT EXISTS `evaluationdataset` (
    `id` VARCHAR(36) NOT NULL PRIMARY KEY,
    `created_at` DATETIME(6) NOT NULL,
    `owner_id` VARCHAR(36) NOT NULL,
    `kb_id` VARCHAR(36) NOT NULL,
    `name` VARCHAR(120) NOT NULL,
    `origin` VARCHAR(30) NOT NULL,
    `cases` JSON NOT NULL,
    KEY `idx_evaluationd_owner_i_402458` (`owner_id`),
    KEY `idx_evaluationd_kb_id_f034e3` (`kb_id`)
) CHARACTER SET utf8mb4;
CREATE TABLE IF NOT EXISTS `evaluationrun` (
    `id` VARCHAR(36) NOT NULL PRIMARY KEY,
    `created_at` DATETIME(6) NOT NULL,
    `owner_id` VARCHAR(36) NOT NULL,
    `kb_id` VARCHAR(36) NOT NULL,
    `status` VARCHAR(20) NOT NULL,
    `results` JSON NOT NULL,
    `metrics` JSON NOT NULL,
    `provider` VARCHAR(30) NOT NULL,
    `error` LONGTEXT NOT NULL,
    KEY `idx_evaluationr_owner_i_43cdf3` (`owner_id`)
) CHARACTER SET utf8mb4;
CREATE TABLE IF NOT EXISTS `feedback` (
    `id` VARCHAR(36) NOT NULL PRIMARY KEY,
    `created_at` DATETIME(6) NOT NULL,
    `owner_id` VARCHAR(36) NOT NULL,
    `run_id` VARCHAR(36) NOT NULL,
    `helpful` BOOL NOT NULL,
    `comment` LONGTEXT NOT NULL,
    UNIQUE KEY `uid_feedback_owner_i_f11a13` (`owner_id`, `run_id`),
    KEY `idx_feedback_owner_i_b8b480` (`owner_id`),
    KEY `idx_feedback_run_id_ee1af4` (`run_id`)
) CHARACTER SET utf8mb4;
CREATE TABLE IF NOT EXISTS `ingestjob` (
    `id` VARCHAR(36) NOT NULL PRIMARY KEY,
    `created_at` DATETIME(6) NOT NULL,
    `owner_id` VARCHAR(36) NOT NULL,
    `doc_id` VARCHAR(36) NOT NULL,
    `kind` VARCHAR(20) NOT NULL,
    `payload` JSON NOT NULL,
    `status` VARCHAR(20) NOT NULL,
    `attempts` INT NOT NULL,
    `lease_owner` VARCHAR(36) NOT NULL,
    `lease_until` DATETIME(6),
    `error` LONGTEXT NOT NULL,
    KEY `idx_ingestjob_doc_id_6b6aab` (`doc_id`),
    KEY `idx_ingestjob_status_fda3a8` (`status`)
) CHARACTER SET utf8mb4;
CREATE TABLE IF NOT EXISTS `knowledgebase` (
    `id` VARCHAR(36) NOT NULL PRIMARY KEY,
    `created_at` DATETIME(6) NOT NULL,
    `owner_id` VARCHAR(36) NOT NULL,
    `name` VARCHAR(120) NOT NULL,
    `description` LONGTEXT NOT NULL,
    `config` JSON NOT NULL,
    `revision` INT NOT NULL,
    KEY `idx_knowledgeba_owner_i_65814a` (`owner_id`)
) CHARACTER SET utf8mb4;
CREATE TABLE IF NOT EXISTS `parentchunk` (
    `id` VARCHAR(36) NOT NULL PRIMARY KEY,
    `created_at` DATETIME(6) NOT NULL,
    `owner_id` VARCHAR(36) NOT NULL,
    `kb_id` VARCHAR(36) NOT NULL,
    `doc_id` VARCHAR(36) NOT NULL,
    `ordinal` INT NOT NULL,
    `content` LONGTEXT NOT NULL,
    `metadata` JSON NOT NULL,
    `edited` BOOL NOT NULL,
    KEY `idx_parentchunk_owner_i_f18727` (`owner_id`),
    KEY `idx_parentchunk_kb_id_9c03d0` (`kb_id`),
    KEY `idx_parentchunk_doc_id_f20ea9` (`doc_id`)
) CHARACTER SET utf8mb4;
CREATE TABLE IF NOT EXISTS `prompttemplate` (
    `id` VARCHAR(36) NOT NULL PRIMARY KEY,
    `created_at` DATETIME(6) NOT NULL,
    `owner_id` VARCHAR(36) NOT NULL,
    `name` VARCHAR(40) NOT NULL,
    `content` LONGTEXT NOT NULL,
    UNIQUE KEY `uid_prompttempl_owner_i_93c7db` (`owner_id`, `name`)
) CHARACTER SET utf8mb4;
CREATE TABLE IF NOT EXISTS `runevent` (
    `id` VARCHAR(36) NOT NULL PRIMARY KEY,
    `created_at` DATETIME(6) NOT NULL,
    `run_id` VARCHAR(36) NOT NULL,
    `sequence` INT NOT NULL,
    `name` VARCHAR(30) NOT NULL,
    `data` JSON NOT NULL,
    UNIQUE KEY `uid_runevent_run_id_ee3969` (`run_id`, `sequence`),
    KEY `idx_runevent_run_id_12b0ac` (`run_id`)
) CHARACTER SET utf8mb4;
CREATE TABLE IF NOT EXISTS `tooldefinition` (
    `id` VARCHAR(36) NOT NULL PRIMARY KEY,
    `created_at` DATETIME(6) NOT NULL,
    `owner_id` VARCHAR(36) NOT NULL,
    `name` VARCHAR(64) NOT NULL,
    `enabled` BOOL NOT NULL,
    UNIQUE KEY `uid_tooldefinit_owner_i_9cbf11` (`owner_id`, `name`)
) CHARACTER SET utf8mb4;
CREATE TABLE IF NOT EXISTS `user` (
    `id` VARCHAR(36) NOT NULL PRIMARY KEY,
    `created_at` DATETIME(6) NOT NULL,
    `username` VARCHAR(64) NOT NULL UNIQUE,
    `password_hash` VARCHAR(256) NOT NULL,
    `is_active` BOOL NOT NULL
) CHARACTER SET utf8mb4;