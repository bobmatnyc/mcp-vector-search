"""Tests for schema version checking and compatibility."""

import json
import tempfile
from pathlib import Path

from mcp_vector_search.core.schema import (
    SCHEMA_VERSION,
    SchemaVersion,
    check_schema_compatibility,
    load_schema_version,
    save_schema_version,
)


def test_schema_version_parsing():
    """Test SchemaVersion parsing and comparison."""
    v1 = SchemaVersion("2.3.4")
    assert v1.major == 2
    assert v1.minor == 3
    assert v1.patch == 4
    assert str(v1) == "2.3.4"


def test_schema_version_compatibility_same_version():
    """Test that same versions are compatible."""
    v1 = SchemaVersion("2.3.4")
    v2 = SchemaVersion("2.3.4")
    assert v1.is_compatible_with(v2)
    assert v2.is_compatible_with(v1)


def test_schema_version_compatibility_minor_upgrade():
    """Test that minor version upgrades are compatible."""
    db_version = SchemaVersion("2.3.0")
    code_version = SchemaVersion("2.4.0")
    # Database 2.3.0 is compatible with code 2.4.0 (code can handle older DB)
    assert db_version.is_compatible_with(code_version)


def test_schema_version_incompatibility_major():
    """Test that major version differences are incompatible."""
    v1 = SchemaVersion("1.0.0")
    v2 = SchemaVersion("2.0.0")
    assert not v1.is_compatible_with(v2)
    assert not v2.is_compatible_with(v1)


def test_schema_version_incompatibility_newer_db():
    """Test that newer database is incompatible with older code."""
    db_version = SchemaVersion("2.5.0")
    code_version = SchemaVersion("2.3.0")
    # Database 2.5.0 is NOT compatible with code 2.3.0 (DB has features code doesn't understand)
    assert not db_version.is_compatible_with(code_version)


def test_save_and_load_schema_version():
    """Test saving and loading schema version."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "lancedb"
        db_path.mkdir(parents=True)

        # Save version
        version = SchemaVersion("2.3.4")
        save_schema_version(db_path, version)

        # Load version
        loaded_version = load_schema_version(db_path)
        assert loaded_version is not None
        assert str(loaded_version) == "2.3.4"

        # Check version file exists in parent directory
        version_file = db_path.parent / "schema_version.json"
        assert version_file.exists()

        # Verify JSON structure
        with open(version_file) as f:
            data = json.load(f)
        assert data["version"] == "2.3.4"
        assert "updated_at" in data


def test_load_missing_schema_version():
    """Test loading schema version when file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "lancedb"
        db_path.mkdir(parents=True)

        loaded_version = load_schema_version(db_path)
        assert loaded_version is None


def test_check_schema_compatibility_no_version():
    """Test compatibility check when no version file exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "lancedb"
        db_path.mkdir(parents=True)

        is_compatible, message = check_schema_compatibility(db_path)
        assert not is_compatible
        assert "No schema version found" in message
        assert "Reset database" in message


def test_check_schema_compatibility_compatible():
    """Test compatibility check when versions are compatible."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "lancedb"
        db_path.mkdir(parents=True)

        # Save current version
        save_schema_version(db_path)

        is_compatible, message = check_schema_compatibility(db_path)
        assert is_compatible
        assert "compatible" in message.lower()


def test_check_schema_compatibility_major_mismatch():
    """Test compatibility check with major version mismatch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "lancedb"
        db_path.mkdir(parents=True)

        # Save old major version
        old_version = SchemaVersion("1.0.0")
        save_schema_version(db_path, old_version)

        is_compatible, message = check_schema_compatibility(db_path)
        assert not is_compatible
        assert "Major version mismatch" in message
        assert "Reset database" in message


def test_check_schema_compatibility_newer_db():
    """Test compatibility check when database is newer than code."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "lancedb"
        db_path.mkdir(parents=True)

        # Parse current version and create a newer one
        current = SchemaVersion(SCHEMA_VERSION)
        newer_version = SchemaVersion(f"{current.major}.{current.minor + 1}.0")
        save_schema_version(db_path, newer_version)

        is_compatible, message = check_schema_compatibility(db_path)
        assert not is_compatible
        assert "newer than code" in message.lower()
        assert "Upgrade mcp-vector-search" in message


def test_schema_version_file_location():
    """Test that schema version is stored in parent directory for lancedb."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Simulate lancedb structure: .mcp-vector-search/lancedb/
        mcp_dir = Path(tmpdir)
        db_path = mcp_dir / "lancedb"
        db_path.mkdir(parents=True)

        save_schema_version(db_path)

        # Version file should be in .mcp-vector-search/ not lancedb/
        version_file = mcp_dir / "schema_version.json"
        assert version_file.exists()

        # Verify it's NOT in the lancedb subdirectory
        wrong_location = db_path / "schema_version.json"
        assert not wrong_location.exists()


def test_default_schema_version_matches_package():
    """Test that default schema version matches package version."""
    from mcp_vector_search import __version__

    assert SCHEMA_VERSION == __version__
