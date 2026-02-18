"""Schema versioning and compatibility checking for MCP Vector Search.

This module ensures that the database schema matches the code version,
preventing infinite retry loops when fields are missing or incompatible.
"""

import json
from datetime import UTC
from pathlib import Path

from loguru import logger

from mcp_vector_search import __version__

# Schema version - ONLY bump when database schema changes (new fields, removed fields)
# This is separate from package __version__ which changes for every release
SCHEMA_VERSION = "2.4.0"  # Last schema change: added git blame fields

# Schema changelog - documents when schema actually changed
SCHEMA_CHANGELOG = {
    "2.4.0": "Added git blame fields (last_author, last_modified, commit_hash)",
    "2.3.0": "Added 'calls' and 'inherits_from' fields to chunks table",
    "2.2.0": "Initial schema with basic chunk fields",
}

# Code versions compatible with each schema version
# All patch versions within the same minor version are compatible
SCHEMA_COMPATIBILITY = {
    "2.4.0": ["2.4.0"],  # Current schema with git blame fields
    "2.3.0": ["2.3.0", "2.3.1", "2.3.2", "2.3.3", "2.3.4", "2.3.5", "2.3.6", "2.3.7"],
    "2.2.0": ["2.2.0", "2.2.1", "2.2.2", "2.2.3", "2.2.4", "2.2.5"],
}

# Required fields for each table/collection
# These are the fields that MUST exist in the database for the current code version
REQUIRED_FIELDS = {
    "chunks": [
        "chunk_id",
        "file_path",
        "content",
        "start_line",
        "end_line",
        "language",
        "chunk_type",
        "function_name",
        "class_name",
        "docstring",
        "imports",
        "calls",  # Added in 2.3.x
        "inherits_from",  # Added in 2.3.x
    ],
    "vectors": [
        "chunk_id",
        "vector",
    ],
}

# Optional fields that can be missing without breaking functionality
OPTIONAL_FIELDS = {
    "chunks": [
        "complexity_score",
        "parent_chunk_id",
        "child_chunk_ids",
        "chunk_depth",
        "decorators",
        "parameters",
        "return_type",
        "type_annotations",
        "subproject_name",
        "subproject_path",
        "nlp_keywords",
        "nlp_code_refs",
        "nlp_technical_terms",
        "last_author",  # Git blame fields (optional - may not be in git)
        "last_modified",
        "commit_hash",
    ],
}


class SchemaVersion:
    """Schema version information and comparison."""

    def __init__(self, version_str: str) -> None:
        """Parse version string (e.g., "2.3.4")."""
        self.version_str = version_str
        parts = version_str.split(".")
        self.major = int(parts[0]) if len(parts) > 0 else 0
        self.minor = int(parts[1]) if len(parts) > 1 else 0
        self.patch = int(parts[2]) if len(parts) > 2 else 0

    def is_compatible_with(self, other: "SchemaVersion") -> bool:
        """Check if two versions are compatible.

        Compatible means:
        - Exact schema version match (same major.minor.patch)
        - Schema versioning is now independent of code versioning
        - Schema only changes when database fields change
        """
        # Exact schema version match required
        return (
            self.major == other.major
            and self.minor == other.minor
            and self.patch == other.patch
        )

    def __str__(self) -> str:
        return self.version_str

    def __repr__(self) -> str:
        return f"SchemaVersion('{self.version_str}')"


def get_schema_version_path(db_path: Path) -> Path:
    """Get path to schema version file.

    Args:
        db_path: Database directory path (e.g., .mcp-vector-search/lancedb)

    Returns:
        Path to schema_version.json
    """
    # Store version in parent directory (.mcp-vector-search)
    mcp_dir = db_path.parent if db_path.name == "lancedb" else db_path
    return mcp_dir / "schema_version.json"


def load_schema_version(db_path: Path) -> SchemaVersion | None:
    """Load schema version from database directory.

    Args:
        db_path: Database directory path

    Returns:
        SchemaVersion if exists, None if not found
    """
    version_file = get_schema_version_path(db_path)

    if not version_file.exists():
        logger.debug(f"No schema version file found at {version_file}")
        return None

    try:
        with open(version_file) as f:
            data = json.load(f)
        version_str = data.get("version", "0.0.0")
        return SchemaVersion(version_str)
    except Exception as e:
        logger.warning(f"Failed to load schema version: {e}")
        return None


def save_schema_version(db_path: Path, version: SchemaVersion | None = None) -> None:
    """Save schema version to database directory.

    Args:
        db_path: Database directory path
        version: Version to save (defaults to current SCHEMA_VERSION)
    """
    if version is None:
        version = SchemaVersion(SCHEMA_VERSION)

    version_file = get_schema_version_path(db_path)

    try:
        # Ensure parent directory exists
        version_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": str(version),
            "updated_at": _get_timestamp(),
        }

        with open(version_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Saved schema version {version} to {version_file}")
    except Exception as e:
        logger.error(f"Failed to save schema version: {e}")


def check_schema_compatibility(db_path: Path) -> tuple[bool, str]:
    """Check if database schema matches current code version.

    Args:
        db_path: Database directory path

    Returns:
        Tuple of (is_compatible, message)
        - is_compatible: True if schema is compatible
        - message: Human-readable status message
    """
    current_version = SchemaVersion(SCHEMA_VERSION)
    db_version = load_schema_version(db_path)

    if db_version is None:
        # No version file - assume old database (pre-2.3.0)
        return (
            False,
            f"No schema version found. Database may be from older version (< 2.3.0).\n"
            f"Current schema version: {current_version}\n"
            f"Database will be automatically reset.",
        )

    if db_version.is_compatible_with(current_version):
        return (
            True,
            f"Schema version {db_version} is compatible (code version: {__version__})",
        )

    # Incompatible schema versions - need reset
    db_changelog = SCHEMA_CHANGELOG.get(str(db_version), "Unknown changes")
    current_changelog = SCHEMA_CHANGELOG.get(str(current_version), "Unknown changes")

    return (
        False,
        f"❌ Schema version mismatch!\n\n"
        f"Database schema: {db_version} ({db_changelog})\n"
        f"Current schema: {current_version} ({current_changelog})\n"
        f"Code version: {__version__}\n\n"
        f"Database schema is incompatible and will be automatically reset.",
    )


def get_missing_fields(db_path: Path) -> dict[str, list[str]]:
    """Get fields that exist in code but not in database.

    This is a placeholder - actual implementation would need to inspect
    the LanceDB/ChromaDB schema to detect missing columns.

    Args:
        db_path: Database directory path

    Returns:
        Dict mapping table names to lists of missing field names
    """
    # For now, return empty dict - field detection requires database inspection
    # This would be implemented by:
    # 1. Opening the LanceDB table
    # 2. Getting schema via table.schema
    # 3. Comparing with REQUIRED_FIELDS
    logger.debug("Field inspection not yet implemented")
    return {}


def _get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    from datetime import datetime

    return datetime.now(UTC).isoformat()


def migrate_schema(db_path: Path, force: bool = False) -> tuple[bool, str]:
    """Attempt to migrate schema to current version.

    For LanceDB, column addition is challenging because:
    - LanceDB uses Apache Arrow schemas which are immutable
    - Adding columns requires recreating the entire table

    Therefore, this function primarily serves to:
    1. Detect incompatibilities
    2. Recommend reset/reindex
    3. Save updated schema version after successful reset

    Args:
        db_path: Database directory path
        force: If True, overwrite existing schema version

    Returns:
        Tuple of (success, message)
    """
    current_version = SchemaVersion(SCHEMA_VERSION)

    if not force:
        is_compatible, msg = check_schema_compatibility(db_path)
        if not is_compatible:
            return (
                False,
                f"Schema migration required:\n{msg}\n\n"
                f"Use --force flag to reset database and reindex.",
            )

    # If we reach here, either force=True or schema is compatible
    save_schema_version(db_path, current_version)
    return (True, f"Schema version updated to {current_version}")
