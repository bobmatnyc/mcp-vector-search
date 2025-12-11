"""SQLite schema definitions for metrics storage.

This module defines the database schema for storing code metrics over time,
enabling trend analysis and historical comparisons.

Schema Design Principles:
    - Normalized structure: Separate tables for files, projects, and smells
    - Temporal tracking: All tables include timestamps for history
    - Efficient queries: Indexes on frequently queried columns
    - Migration support: Schema version tracking for future changes

Tables:
    - schema_version: Track schema version for migrations
    - file_metrics: Per-file metrics snapshots
    - project_snapshots: Project-wide metric aggregates
    - code_smells: Detected code smell instances

Performance Considerations:
    - Composite indexes for common query patterns
    - Foreign keys for referential integrity
    - Cascading deletes for cleanup

Design Decision: SQLite for Simplicity
    Rationale: Chosen SQLite over PostgreSQL/MySQL for:
    - Zero configuration (no server setup)
    - Single-file portability
    - Adequate performance for ~10K files
    - Built into Python standard library

    Trade-offs:
    - Concurrency: Limited to single writer (acceptable for CLI tool)
    - Scale: Suitable for projects up to ~10K files
    - Features: No advanced types (JSON columns, etc.)

    Extension Points: Schema includes version tracking to enable migration
    to PostgreSQL if concurrent access or >10K file scale becomes necessary.
"""

# Schema version for migration tracking
SCHEMA_VERSION = "1.0"

# SQL statement to create schema version table
CREATE_SCHEMA_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_version (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# SQL statement to initialize schema version
INIT_SCHEMA_VERSION = f"""
INSERT OR IGNORE INTO schema_version (version) VALUES ('{SCHEMA_VERSION}');
"""

# SQL statement to create file_metrics table
CREATE_FILE_METRICS_TABLE = """
CREATE TABLE IF NOT EXISTS file_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    project_id INTEGER NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Line counts
    total_lines INTEGER NOT NULL DEFAULT 0,
    code_lines INTEGER NOT NULL DEFAULT 0,
    comment_lines INTEGER NOT NULL DEFAULT 0,
    blank_lines INTEGER NOT NULL DEFAULT 0,

    -- Structure counts
    function_count INTEGER NOT NULL DEFAULT 0,
    class_count INTEGER NOT NULL DEFAULT 0,
    method_count INTEGER NOT NULL DEFAULT 0,

    -- Complexity metrics
    cognitive_complexity INTEGER NOT NULL DEFAULT 0,
    cyclomatic_complexity INTEGER NOT NULL DEFAULT 0,
    total_complexity INTEGER NOT NULL DEFAULT 0,
    avg_complexity REAL NOT NULL DEFAULT 0.0,
    max_complexity INTEGER NOT NULL DEFAULT 0,

    -- Code quality
    smell_count INTEGER NOT NULL DEFAULT 0,
    health_score REAL NOT NULL DEFAULT 1.0,
    complexity_grade TEXT NOT NULL DEFAULT 'A',

    -- Foreign key to project snapshot
    FOREIGN KEY (project_id) REFERENCES project_snapshots(id) ON DELETE CASCADE,

    -- Ensure file_path + project_id + timestamp is unique
    UNIQUE(file_path, project_id, timestamp)
);
"""

# SQL statements to create indexes on file_metrics (separate statements)
CREATE_FILE_METRICS_INDEXES = [
    """CREATE INDEX IF NOT EXISTS idx_file_metrics_file_path
        ON file_metrics(file_path)""",
    """CREATE INDEX IF NOT EXISTS idx_file_metrics_project_id
        ON file_metrics(project_id)""",
    """CREATE INDEX IF NOT EXISTS idx_file_metrics_timestamp
        ON file_metrics(timestamp DESC)""",
    """CREATE INDEX IF NOT EXISTS idx_file_metrics_complexity
        ON file_metrics(avg_complexity DESC)""",
    """CREATE INDEX IF NOT EXISTS idx_file_metrics_health
        ON file_metrics(health_score ASC)""",
]

# SQL statement to create project_snapshots table
CREATE_PROJECT_SNAPSHOTS_TABLE = """
CREATE TABLE IF NOT EXISTS project_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_path TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Project totals
    total_files INTEGER NOT NULL DEFAULT 0,
    total_lines INTEGER NOT NULL DEFAULT 0,
    total_functions INTEGER NOT NULL DEFAULT 0,
    total_classes INTEGER NOT NULL DEFAULT 0,

    -- Complexity aggregates
    avg_complexity REAL NOT NULL DEFAULT 0.0,
    max_complexity INTEGER NOT NULL DEFAULT 0,
    total_complexity INTEGER NOT NULL DEFAULT 0,

    -- Code quality
    total_smells INTEGER NOT NULL DEFAULT 0,
    avg_health_score REAL NOT NULL DEFAULT 1.0,

    -- Grade distribution (JSON-encoded dict)
    grade_distribution TEXT NOT NULL DEFAULT '{}',

    -- Git metadata for traceability
    git_commit TEXT,
    git_branch TEXT,
    git_remote TEXT,

    -- Tool version for compatibility
    tool_version TEXT,

    -- Ensure project_path + timestamp is unique
    UNIQUE(project_path, timestamp)
);
"""

# SQL statements to create indexes on project_snapshots (separate statements)
CREATE_PROJECT_SNAPSHOTS_INDEXES = [
    """CREATE INDEX IF NOT EXISTS idx_project_snapshots_project_path
        ON project_snapshots(project_path)""",
    """CREATE INDEX IF NOT EXISTS idx_project_snapshots_timestamp
        ON project_snapshots(timestamp DESC)""",
    """CREATE INDEX IF NOT EXISTS idx_project_snapshots_complexity
        ON project_snapshots(avg_complexity DESC)""",
    """CREATE INDEX IF NOT EXISTS idx_project_snapshots_git_commit
        ON project_snapshots(git_commit)""",
]

# SQL statement to create code_smells table
CREATE_CODE_SMELLS_TABLE = """
CREATE TABLE IF NOT EXISTS code_smells (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    snapshot_id INTEGER NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Smell classification
    smell_type TEXT NOT NULL,
    severity TEXT NOT NULL CHECK(severity IN ('low', 'medium', 'high', 'critical')),

    -- Location in file (line number or range)
    location TEXT,

    -- Metric that triggered the smell
    metric_name TEXT,
    metric_value REAL,
    threshold REAL,

    -- Foreign keys
    FOREIGN KEY (file_id) REFERENCES file_metrics(id) ON DELETE CASCADE,
    FOREIGN KEY (snapshot_id) REFERENCES project_snapshots(id) ON DELETE CASCADE
);
"""

# SQL statements to create indexes on code_smells (separate statements)
CREATE_CODE_SMELLS_INDEXES = [
    """CREATE INDEX IF NOT EXISTS idx_code_smells_file_id
        ON code_smells(file_id)""",
    """CREATE INDEX IF NOT EXISTS idx_code_smells_snapshot_id
        ON code_smells(snapshot_id)""",
    """CREATE INDEX IF NOT EXISTS idx_code_smells_smell_type
        ON code_smells(smell_type)""",
    """CREATE INDEX IF NOT EXISTS idx_code_smells_severity
        ON code_smells(severity)""",
    """CREATE INDEX IF NOT EXISTS idx_code_smells_timestamp
        ON code_smells(timestamp DESC)""",
]

# Complete schema initialization (all CREATE statements)
# Flatten all statements into a single list
INIT_SCHEMA_SQL = (
    [CREATE_SCHEMA_VERSION_TABLE, INIT_SCHEMA_VERSION]
    + [CREATE_PROJECT_SNAPSHOTS_TABLE]
    + CREATE_PROJECT_SNAPSHOTS_INDEXES
    + [CREATE_FILE_METRICS_TABLE]
    + CREATE_FILE_METRICS_INDEXES
    + [CREATE_CODE_SMELLS_TABLE]
    + CREATE_CODE_SMELLS_INDEXES
)


def get_schema_version_query() -> str:
    """Get SQL query to retrieve current schema version.

    Returns:
        SQL SELECT statement to fetch schema version
    """
    return "SELECT version FROM schema_version ORDER BY applied_at DESC LIMIT 1"


def get_migration_queries(from_version: str, to_version: str) -> list[str]:
    """Get migration SQL statements for schema upgrade.

    Args:
        from_version: Current schema version
        to_version: Target schema version

    Returns:
        List of SQL statements to apply migration

    Future Extension: When schema changes, add migration logic here.
    Example:
        if from_version == "1.0" and to_version == "2.0":
            return [
                "ALTER TABLE file_metrics ADD COLUMN new_field INTEGER DEFAULT 0",
                "UPDATE schema_version SET version = '2.0'",
            ]
    """
    # Currently only version 1.0 exists, no migrations yet
    if from_version == to_version:
        return []

    raise ValueError(
        f"No migration path from version {from_version} to {to_version}. "
        f"Current schema version: {SCHEMA_VERSION}"
    )
