-- ==========================================================================
-- MRPL Sovereign AI Workbench — SQLite Schema
-- All tables for the MVP application database.
-- Designed for WAL mode + foreign key enforcement.
-- ==========================================================================

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ==========================================================================
-- 1. ROLES
-- ==========================================================================
CREATE TABLE IF NOT EXISTS roles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,          -- e.g. administrator, engineer, document_manager, auditor, viewer
    description TEXT    NOT NULL DEFAULT '',
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- ==========================================================================
-- 2. USERS
-- ==========================================================================
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT    NOT NULL UNIQUE,
    display_name    TEXT    NOT NULL,
    password_hash   TEXT    NOT NULL,             -- Argon2id hash
    clearance       TEXT    NOT NULL DEFAULT 'INTERNAL', -- PUBLIC, INTERNAL, CONFIDENTIAL, HIGHLY_CONFIDENTIAL
    is_active       INTEGER NOT NULL DEFAULT 1,  -- 0 = deactivated
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);

-- ==========================================================================
-- 3. USER ↔ ROLE join table (many-to-many)
-- ==========================================================================
CREATE TABLE IF NOT EXISTS user_roles (
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role_id     INTEGER NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    assigned_at TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    PRIMARY KEY (user_id, role_id)
);

CREATE INDEX IF NOT EXISTS idx_user_roles_user ON user_roles(user_id);

-- ==========================================================================
-- 4. SESSIONS (auth tokens)
-- ==========================================================================
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token       TEXT    NOT NULL UNIQUE,          -- secrets.token_urlsafe(48)
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    expires_at  TEXT    NOT NULL,                 -- ISO-8601 UTC
    is_active   INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token);
CREATE INDEX IF NOT EXISTS idx_sessions_user  ON sessions(user_id);

-- ==========================================================================
-- 5. CHAT SESSIONS
-- ==========================================================================
CREATE TABLE IF NOT EXISTS chat_sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title       TEXT    NOT NULL DEFAULT 'New Chat',
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_user ON chat_sessions(user_id);

-- ==========================================================================
-- 6. CHAT MESSAGES
-- ==========================================================================
CREATE TABLE IF NOT EXISTS chat_messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role            TEXT    NOT NULL CHECK(role IN ('user', 'assistant', 'system', 'tool')),
    content         TEXT    NOT NULL,
    model_id        TEXT,                          -- which model produced this (NULL for user messages)
    tool_calls      TEXT,                          -- JSON array of tool calls, if any
    sources         TEXT,                          -- JSON array of RAG source citations
    execution_ms    INTEGER,                       -- how long the LLM took to respond
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id);

-- ==========================================================================
-- 7. TASKS (agent task tracking)
-- ==========================================================================
CREATE TABLE IF NOT EXISTS tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id  INTEGER REFERENCES chat_sessions(id) ON DELETE SET NULL,
    task_type   TEXT    NOT NULL,                  -- from model_registry task_type_routing
    status      TEXT    NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    input_data  TEXT    NOT NULL,                  -- JSON: original request + context
    output_data TEXT,                              -- JSON: final result
    error_info  TEXT,                              -- JSON: error details if failed
    model_id    TEXT,                              -- model used
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_tasks_user   ON tasks(user_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);

-- ==========================================================================
-- 8. TASK STEPS (individual steps within an agent task)
-- ==========================================================================
CREATE TABLE IF NOT EXISTS task_steps (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    step_index  INTEGER NOT NULL,                  -- ordering within the task
    state_name  TEXT    NOT NULL,                   -- LangGraph node name
    input_data  TEXT,                               -- JSON: state entering this step
    output_data TEXT,                               -- JSON: state exiting this step
    duration_ms INTEGER,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_task_steps_task ON task_steps(task_id);

-- ==========================================================================
-- 9. WORKFLOW RUNS (n8n integration tracking)
-- ==========================================================================
CREATE TABLE IF NOT EXISTS workflow_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    workflow_name   TEXT    NOT NULL,               -- human-readable name
    n8n_execution_id TEXT,                          -- n8n's execution ID
    status          TEXT    NOT NULL DEFAULT 'triggered' CHECK(status IN ('triggered', 'running', 'completed', 'failed')),
    trigger_source  TEXT    NOT NULL,               -- e.g. 'api', 'schedule', 'manual'
    input_data      TEXT,                           -- JSON
    output_data     TEXT,                           -- JSON
    error_info      TEXT,
    triggered_by    INTEGER REFERENCES users(id) ON DELETE SET NULL,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    completed_at    TEXT
);

-- ==========================================================================
-- 10. AUDIT LOGS (append-only)
-- ==========================================================================
CREATE TABLE IF NOT EXISTS audit_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER REFERENCES users(id) ON DELETE SET NULL,
    username    TEXT,                               -- denormalized for log survival
    action      TEXT    NOT NULL,                   -- e.g. 'login', 'chat_message', 'file_upload'
    target      TEXT,                               -- what was acted on (e.g. 'chat_session:42')
    outcome     TEXT    NOT NULL CHECK(outcome IN ('success', 'failure', 'denied')),
    details     TEXT,                               -- JSON with extra context
    ip_address  TEXT,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_audit_logs_user    ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action  ON audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_audit_logs_outcome ON audit_logs(outcome);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created ON audit_logs(created_at);

-- ==========================================================================
-- 11. MODEL REGISTRY (runtime mirror of model_registry.yaml)
-- ==========================================================================
CREATE TABLE IF NOT EXISTS model_registry (
    id                  TEXT    PRIMARY KEY,        -- e.g. 'gemma3_4b'
    display_name        TEXT    NOT NULL,
    provider            TEXT    NOT NULL,
    ollama_model_name   TEXT    NOT NULL,
    endpoint            TEXT    NOT NULL,
    modality            TEXT    NOT NULL,           -- JSON array: ["text", "vision"]
    max_context_window  INTEGER NOT NULL,
    runtime_context_window INTEGER NOT NULL,
    vram_gb_estimate    REAL,
    enabled             INTEGER NOT NULL DEFAULT 1,
    task_tags           TEXT    NOT NULL,           -- JSON array
    priority            INTEGER NOT NULL DEFAULT 1,
    notes               TEXT,
    synced_at           TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- ==========================================================================
-- 12. UPLOADED FILES
-- ==========================================================================
CREATE TABLE IF NOT EXISTS uploaded_files (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    original_name   TEXT    NOT NULL,
    stored_path     TEXT    NOT NULL,               -- path on disk
    file_size_bytes INTEGER NOT NULL,
    mime_type       TEXT,
    category        TEXT,                           -- KB category if classified
    classification  TEXT    NOT NULL DEFAULT 'INTERNAL',
    department      TEXT    NOT NULL DEFAULT 'GENERAL',
    owner           TEXT    NOT NULL DEFAULT 'SYSTEM',
    version         TEXT    NOT NULL DEFAULT '1.0',
    doc_id          TEXT,
    is_indexed      INTEGER NOT NULL DEFAULT 0,     -- 1 = indexed into ChromaDB
    index_status    TEXT    DEFAULT 'pending' CHECK(index_status IN ('pending', 'indexing', 'indexed', 'completed', 'failed')),
    error_message   TEXT,
    ocr_applied     INTEGER NOT NULL DEFAULT 0,
    uploaded_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_uploaded_files_user ON uploaded_files(user_id);

-- ==========================================================================
-- 13. GENERATED FILES
-- ==========================================================================
CREATE TABLE IF NOT EXISTS generated_files (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    task_id         INTEGER REFERENCES tasks(id) ON DELETE SET NULL,
    file_type       TEXT    NOT NULL CHECK(file_type IN ('docx', 'xlsx', 'pptx', 'pdf', 'csv', 'txt', 'other')),
    original_name   TEXT    NOT NULL,
    stored_path     TEXT    NOT NULL,
    file_size_bytes INTEGER,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_generated_files_user ON generated_files(user_id);

-- ==========================================================================
-- 14. SYSTEM EVENTS
-- ==========================================================================
CREATE TABLE IF NOT EXISTS system_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type  TEXT    NOT NULL,                   -- e.g. 'startup', 'shutdown', 'model_load', 'error'
    severity    TEXT    NOT NULL DEFAULT 'info' CHECK(severity IN ('info', 'warning', 'error', 'critical')),
    message     TEXT    NOT NULL,
    details     TEXT,                               -- JSON
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_system_events_type    ON system_events(event_type);
CREATE INDEX IF NOT EXISTS idx_system_events_created ON system_events(created_at);

-- ==========================================================================
-- 15. FEATURE STATUS
-- ==========================================================================
CREATE TABLE IF NOT EXISTS feature_status (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_key TEXT    NOT NULL UNIQUE,            -- e.g. 'rag_search', 'code_sandbox'
    display_name TEXT   NOT NULL,
    category    TEXT    NOT NULL,                   -- grouping: 'core', 'tools', 'admin', 'integration'
    status      TEXT    NOT NULL DEFAULT 'planned' CHECK(status IN ('implemented', 'in_progress', 'planned', 'unavailable')),
    notes       TEXT,
    updated_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_feature_status_key ON feature_status(feature_key);

-- ==========================================================================
-- TRIGGERS: auto-update updated_at timestamps
-- ==========================================================================
CREATE TRIGGER IF NOT EXISTS trg_users_updated_at
AFTER UPDATE ON users
FOR EACH ROW
BEGIN
    UPDATE users SET updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_chat_sessions_updated_at
AFTER UPDATE ON chat_sessions
FOR EACH ROW
BEGIN
    UPDATE chat_sessions SET updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_feature_status_updated_at
AFTER UPDATE ON feature_status
FOR EACH ROW
BEGIN
    UPDATE feature_status SET updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = OLD.id;
END;
