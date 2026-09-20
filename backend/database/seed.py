"""
First-run database seeder.

Creates tables, seeds default roles, creates the default admin user,
syncs the model registry from YAML, and populates initial feature status.
"""

import logging

from backend.database.connection import configure, execute_schema, get_connection
from backend.database.repositories import users, models as models_repo

logger = logging.getLogger(__name__)

# Default roles for the MRPL Sovereign AI Workbench.
DEFAULT_ROLES = [
    ("administrator", "Full system access. User management, model configuration, audit log access."),
    ("engineer", "Chat, RAG queries, code sandbox, document generation."),
    ("reviewer", "Chat, view documents/results, review AI outputs, approve/reject high-risk actions."),
    ("document_manager", "Upload, index, and manage knowledge base documents."),
    ("auditor", "Read-only access to audit logs and system status."),
    ("viewer", "Read-only chat access. Cannot upload or generate files."),
]

import os

# Default admin credentials — configurable via environment; marked for rotation
DEFAULT_ADMIN_USERNAME = os.getenv("MRPL_ADMIN_USERNAME", "admin")
DEFAULT_ADMIN_DISPLAY_NAME = "[DEMO] System Administrator"
DEFAULT_ADMIN_PASSWORD = os.getenv("MRPL_ADMIN_PASSWORD", "1234567890")
DEFAULT_DEMO_PASSWORD = os.getenv("MRPL_DEMO_PASSWORD", "1234567890")

# Feature status entries — all MVP features and their initial status.
INITIAL_FEATURES = [
    # Core
    ("authentication", "User Authentication", "core", "implemented"),
    ("rbac", "Role-Based Access Control", "core", "implemented"),
    ("session_management", "Session Management", "core", "implemented"),
    ("chat_persistence", "Chat History Persistence", "core", "implemented"),
    ("audit_logging", "Audit Logging", "core", "implemented"),
    # Tools
    ("rag_search", "RAG Knowledge Base Search", "tools", "implemented"),
    ("ocr_extraction", "OCR Text Extraction", "tools", "implemented"),
    ("code_sandbox", "Code Sandbox Execution", "tools", "implemented"),
    ("docgen_docx", "Word Document Generation", "tools", "implemented"),
    ("docgen_xlsx", "Excel Document Generation", "tools", "implemented"),
    ("docgen_pptx", "PowerPoint Generation", "tools", "implemented"),
    # Agent
    ("agent_orchestrator", "LangGraph Agent Orchestrator", "agent", "implemented"),
    ("task_routing", "Intelligent Task Routing", "agent", "implemented"),
    ("multi_step_reasoning", "Multi-Step Reasoning", "agent", "implemented"),
    # Admin
    ("user_management", "User Management", "admin", "implemented"),
    ("model_registry", "Model Registry", "admin", "implemented"),
    ("system_monitoring", "System Status Monitoring", "admin", "implemented"),
    ("feature_status", "Feature Status Dashboard", "admin", "implemented"),
    # Integration
    ("n8n_workflows", "n8n Workflow Integration", "integration", "planned"),
    ("network_monitoring", "Network Air-Gap Monitoring", "integration", "implemented"),
    # Frontend
    ("web_ui", "React Web Interface", "frontend", "implemented"),
    ("dark_theme", "Dark Theme with Glassmorphism", "frontend", "implemented"),
]


def seed_database(db_path=None) -> None:
    """
    Run the full seed sequence. Idempotent — safe to run multiple times.
    """
    configure(db_path)
    execute_schema()

    logger.info("Seeding database...")

    _seed_roles()
    _seed_admin_user()
    _seed_default_users()
    _seed_model_registry()
    _seed_feature_status()

    logger.info("Database seeding complete.")


def _seed_roles() -> None:
    """Create default roles if they don't exist."""
    for name, description in DEFAULT_ROLES:
        existing = users.get_role_by_name(name)
        if existing is None:
            users.create_role(name, description)
            logger.info("Seeded role: %s", name)
        else:
            logger.debug("Role '%s' already exists, skipping", name)


def _seed_admin_user() -> None:
    """Create the default admin user if it doesn't exist."""
    existing = users.get_user_by_username(DEFAULT_ADMIN_USERNAME)
    if existing is not None:
        logger.debug("Admin user '%s' already exists, skipping", DEFAULT_ADMIN_USERNAME)
        return

    user_id = users.create_user(
        username=DEFAULT_ADMIN_USERNAME,
        display_name=DEFAULT_ADMIN_DISPLAY_NAME,
        password=DEFAULT_ADMIN_PASSWORD,
    )
    users.assign_role(user_id, "administrator")
    logger.info(
        "Seeded admin user '%s' (id=%d) with role 'administrator'. "
        "DEFAULT PASSWORD — MUST BE CHANGED ON FIRST LOGIN.",
        DEFAULT_ADMIN_USERNAME,
        user_id,
    )


def _seed_default_users() -> None:
    """Seed sample role users for prototype demonstration: engineer, reviewer, viewer."""
    sample_users = [
        ("engineer", "[DEMO] Refinery Process Engineer", DEFAULT_DEMO_PASSWORD, "engineer", "CONFIDENTIAL"),
        ("reviewer", "[DEMO] Refinery Safety Reviewer", DEFAULT_DEMO_PASSWORD, "reviewer", "CONFIDENTIAL"),
        ("viewer", "[DEMO] Guest Viewer", DEFAULT_DEMO_PASSWORD, "viewer", "PUBLIC"),
    ]
    for username, display_name, password, role, clearance in sample_users:
        existing = users.get_user_by_username(username)
        if existing is None:
            uid = users.create_user(
                username=username,
                display_name=display_name,
                password=password,
                clearance=clearance,
            )
            users.assign_role(uid, role)
            logger.info(
                "Seeded DEMO user '%s' with role '%s' and clearance '%s'. (Rotate credentials prior to production).",
                username,
                role,
                clearance,
            )


def _seed_model_registry() -> None:
    """Sync models from model_registry.yaml into SQLite."""
    try:
        count = models_repo.sync_from_yaml()
        logger.info("Synced %d models from YAML registry", count)
    except FileNotFoundError as exc:
        logger.warning("Model registry YAML not found: %s", exc)
    except Exception as exc:
        logger.error("Failed to sync model registry: %s", exc)
        raise


def _seed_feature_status() -> None:
    """Populate feature_status table if entries don't exist."""
    conn = get_connection()

    for feature_key, display_name, category, status in INITIAL_FEATURES:
        existing = conn.execute(
            "SELECT id FROM feature_status WHERE feature_key = ?",
            (feature_key,),
        ).fetchone()

        if existing is None:
            conn.execute(
                """
                INSERT INTO feature_status (feature_key, display_name, category, status)
                VALUES (?, ?, ?, ?)
                """,
                (feature_key, display_name, category, status),
            )
            logger.debug("Seeded feature status: %s", feature_key)

    conn.commit()
    logger.info("Seeded %d feature status entries", len(INITIAL_FEATURES))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    seed_database()
    print("Database seeded successfully.")
