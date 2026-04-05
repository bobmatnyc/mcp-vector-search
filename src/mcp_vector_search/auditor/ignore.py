"""Ignore list management for the privacy-policy auditor.

Task 3: Loads .audit-ignore.yml from the target repo, validates entries,
and provides matching logic for PolicyClaim instances.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import yaml
from loguru import logger
from pydantic import ValidationError

from .models import AuditIgnoreEntry, PolicyClaim


class IgnoreList:
    """Parsed and validated ignore list from .audit-ignore.yml.

    Usage:
        ignore_list = IgnoreList.load(target_repo_path)
        entry = ignore_list.matches(claim)
        if entry:
            # claim is ignored
    """

    IGNORE_FILE_NAME = ".audit-ignore.yml"

    def __init__(self, entries: list[AuditIgnoreEntry]) -> None:
        self._entries = entries

    @classmethod
    def load(cls, target_repo: Path) -> IgnoreList:
        """Load and validate .audit-ignore.yml from the target repo.

        Args:
            target_repo: Root directory of the repository being audited.

        Returns:
            IgnoreList instance (may be empty if no file found).
        """
        ignore_file = target_repo / cls.IGNORE_FILE_NAME
        if not ignore_file.exists():
            logger.debug(
                "No %s found in %s — ignore list is empty",
                cls.IGNORE_FILE_NAME,
                target_repo,
            )
            return cls([])

        try:
            raw = yaml.safe_load(ignore_file.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            logger.warning(
                "Failed to parse %s: %s — ignoring file",
                ignore_file,
                exc,
            )
            return cls([])

        if not isinstance(raw, list):
            logger.warning(
                "%s must contain a YAML list at the top level — ignoring file",
                ignore_file,
            )
            return cls([])

        today = date.today()
        valid_entries: list[AuditIgnoreEntry] = []

        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                logger.warning("Ignore entry #%d is not a mapping — skipping", i)
                continue

            try:
                entry = AuditIgnoreEntry.model_validate(item)
            except ValidationError as exc:
                logger.warning(
                    "Ignore entry #%d failed validation (justification too short?): %s",
                    i,
                    exc,
                )
                continue

            # Warn on expired entries but do NOT include them
            if entry.expires is not None and entry.expires < today:
                logger.warning(
                    "Ignore entry #%d (claim_id=%s) expired on %s — skipping",
                    i,
                    entry.claim_id,
                    entry.expires,
                )
                continue

            valid_entries.append(entry)

        logger.debug(
            "Loaded %d valid ignore entries from %s",
            len(valid_entries),
            ignore_file,
        )
        return cls(valid_entries)

    def matches(self, claim: PolicyClaim) -> AuditIgnoreEntry | None:
        """Return the first matching ignore entry for a claim, or None.

        Matching rules (checked in order):
        1. claim_id match (exact)
        2. category match (exact)
        3. pattern match (regex against claim.normalized)

        Args:
            claim: The PolicyClaim to check.

        Returns:
            The first matching AuditIgnoreEntry, or None if no match.
        """
        for entry in self._entries:
            if entry.claim_id is not None and entry.claim_id == claim.id:
                return entry
            if entry.category is not None and entry.category == claim.category:
                return entry
            if entry.pattern is not None:
                try:
                    if re.search(entry.pattern, claim.normalized, re.IGNORECASE):
                        return entry
                except re.error as exc:
                    logger.warning(
                        "Invalid regex pattern in ignore entry '%s': %s",
                        entry.pattern,
                        exc,
                    )
        return None

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)
