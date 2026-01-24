"""Base migration interface and data models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class MigrationStatus(str, Enum):
    """Status of a migration execution."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class MigrationContext:
    """Context provided to migrations during execution."""

    project_root: Path
    index_path: Path
    config: dict[str, Any]
    dry_run: bool = False

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value with fallback."""
        return self.config.get(key, default)


@dataclass
class MigrationResult:
    """Result of a migration execution."""

    migration_id: str
    version: str
    name: str
    status: MigrationStatus
    message: str = ""
    executed_at: datetime | None = None
    duration_seconds: float = 0.0
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for storage."""
        return {
            "migration_id": self.migration_id,
            "version": self.version,
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "executed_at": (self.executed_at.isoformat() if self.executed_at else None),
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MigrationResult":
        """Create result from stored dictionary."""
        return cls(
            migration_id=data["migration_id"],
            version=data["version"],
            name=data["name"],
            status=MigrationStatus(data["status"]),
            message=data.get("message", ""),
            executed_at=(
                datetime.fromisoformat(data["executed_at"])
                if data.get("executed_at")
                else None
            ),
            duration_seconds=data.get("duration_seconds", 0.0),
            metadata=data.get("metadata"),
        )


class Migration(ABC):
    """Abstract base class for migrations.

    Each migration represents a specific version upgrade that may require
    data migration, index rebuilding, or configuration changes.
    """

    # Migration metadata (override in subclasses)
    version: str = "0.0.0"
    name: str = "base_migration"
    description: str = "Base migration class"

    @property
    def migration_id(self) -> str:
        """Unique identifier for this migration."""
        return f"{self.version}_{self.name}"

    @abstractmethod
    def check_needed(self, context: MigrationContext) -> bool:
        """Check if this migration needs to run.

        Args:
            context: Migration context with project info

        Returns:
            True if migration should execute, False to skip
        """
        pass

    @abstractmethod
    def execute(self, context: MigrationContext) -> MigrationResult:
        """Execute the migration.

        Args:
            context: Migration context with project info

        Returns:
            Result of migration execution
        """
        pass

    def rollback(self, context: MigrationContext) -> bool:
        """Optionally rollback the migration.

        Args:
            context: Migration context with project info

        Returns:
            True if rollback successful, False otherwise
        """
        return False

    def __repr__(self) -> str:
        """String representation of migration."""
        return f"<Migration {self.migration_id}: {self.description}>"
