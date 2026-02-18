"""Integration test for automatic schema reset on mismatch."""

import json
import shutil
import tempfile
from pathlib import Path

from mcp_vector_search.core.schema import (
    SCHEMA_VERSION,
    SchemaVersion,
    check_schema_compatibility,
    save_schema_version,
)


def test_auto_reset_workflow():
    """Test the full auto-reset workflow when schema version mismatches.

    This simulates what happens in index.py when a schema mismatch is detected.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup: Create database with old schema version
        db_path = Path(tmpdir) / "lancedb"
        db_path.mkdir(parents=True)

        # Save an old schema version (simulating existing database)
        old_version = SchemaVersion("2.2.0")
        save_schema_version(db_path, old_version)

        # Create some fake data to simulate existing database
        fake_data_file = db_path / "data.lance"
        fake_data_file.write_text("old database data")

        # Verify old database exists
        assert fake_data_file.exists()
        assert load_old_schema_version(db_path) == "2.2.0"

        # Step 1: Check compatibility (should fail)
        is_compatible, message = check_schema_compatibility(db_path)
        assert not is_compatible
        assert "Schema version mismatch" in message
        assert "automatically reset" in message

        # Step 2: Auto-reset (simulate what index.py does)
        if not is_compatible:
            # Remove old database
            if db_path.exists():
                shutil.rmtree(db_path)

            # Recreate directory
            db_path.mkdir(parents=True, exist_ok=True)

            # Save new schema version
            save_schema_version(db_path)

        # Step 3: Verify reset succeeded
        # Old data should be gone
        assert not fake_data_file.exists()

        # New schema version should be saved
        new_version = load_old_schema_version(db_path)
        assert new_version == SCHEMA_VERSION

        # Compatibility check should now pass
        is_compatible, message = check_schema_compatibility(db_path)
        assert is_compatible
        assert "compatible" in message.lower()


def test_auto_reset_no_version_file():
    """Test auto-reset when no schema version file exists (old database)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup: Create database without schema version (pre-2.3.0)
        db_path = Path(tmpdir) / "lancedb"
        db_path.mkdir(parents=True)

        # Create fake data
        fake_data_file = db_path / "data.lance"
        fake_data_file.write_text("old database data")

        # Step 1: Check compatibility (should fail - no version file)
        is_compatible, message = check_schema_compatibility(db_path)
        assert not is_compatible
        assert "No schema version found" in message

        # Step 2: Auto-reset
        if not is_compatible:
            if db_path.exists():
                shutil.rmtree(db_path)
            db_path.mkdir(parents=True, exist_ok=True)
            save_schema_version(db_path)

        # Step 3: Verify reset succeeded
        assert not fake_data_file.exists()

        # New schema version should be saved
        new_version = load_old_schema_version(db_path)
        assert new_version == SCHEMA_VERSION

        # Compatibility check should now pass
        is_compatible, message = check_schema_compatibility(db_path)
        assert is_compatible


def test_no_reset_when_schema_compatible():
    """Test that no reset occurs when schema is already compatible."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup: Create database with current schema version
        db_path = Path(tmpdir) / "lancedb"
        db_path.mkdir(parents=True)

        # Save current schema version
        save_schema_version(db_path)

        # Create fake data
        fake_data_file = db_path / "data.lance"
        fake_data_file.write_text("current database data")

        # Step 1: Check compatibility (should pass)
        is_compatible, message = check_schema_compatibility(db_path)
        assert is_compatible

        # Step 2: No reset needed
        # Data should still exist
        assert fake_data_file.exists()
        assert fake_data_file.read_text() == "current database data"

        # Schema version should be unchanged
        current_version = load_old_schema_version(db_path)
        assert current_version == SCHEMA_VERSION


def load_old_schema_version(db_path: Path) -> str | None:
    """Helper to load schema version from file."""
    version_file = db_path.parent / "schema_version.json"
    if not version_file.exists():
        return None

    with open(version_file) as f:
        data = json.load(f)
    return data.get("version")
