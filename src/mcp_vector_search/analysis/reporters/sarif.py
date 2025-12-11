"""SARIF 2.1.0 output format for code analysis results.

This module provides SARIF (Static Analysis Results Interchange Format) 2.1.0
compliant output for code smells and structural analysis results. SARIF is an
OASIS standard format for sharing static analysis results between tools.

SARIF Specification: https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html

Example:
    >>> from pathlib import Path
    >>> from ..collectors.smells import CodeSmell, SmellSeverity
    >>> reporter = SARIFReporter()
    >>> smells = [CodeSmell(...)]
    >>> reporter.write_sarif(smells, Path("report.sarif"))
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..collectors.smells import CodeSmell, SmellSeverity


@dataclass
class SARIFReporter:
    """Generate SARIF 2.1.0 formatted reports for code analysis.

    SARIF (Static Analysis Results Interchange Format) is an industry-standard
    JSON-based format for static analysis tool output, enabling integration with
    IDEs, CI/CD systems, and security tools.

    Attributes:
        tool_name: Name of the analysis tool (default: "MCP Vector Search")
        tool_version: Version of the tool (default: from package)
        tool_uri: URI to tool documentation/homepage
        include_help_text: Include help text for each rule (default: True)
        include_fingerprints: Include result fingerprints for deduplication (default: True)

    Example:
        >>> reporter = SARIFReporter()
        >>> sarif_doc = reporter.generate_sarif(code_smells, base_path=Path("/project"))
        >>> reporter.write_sarif(code_smells, Path("report.sarif"))
    """

    tool_name: str = "MCP Vector Search"
    tool_version: str = "1.0.3"
    tool_uri: str = "https://github.com/bobmatnyc/mcp-vector-search"
    include_help_text: bool = True
    include_fingerprints: bool = True

    def generate_sarif(
        self, smells: list[CodeSmell], base_path: Path | None = None
    ) -> dict[str, Any]:
        """Generate SARIF 2.1.0 document from code smells.

        Creates a complete SARIF document with tool metadata, rules, and results.
        All file paths are made relative to base_path if provided.

        Args:
            smells: List of detected code smells to report
            base_path: Base directory for making paths relative (optional)
                      If None, uses absolute paths

        Returns:
            Dictionary containing SARIF 2.1.0 compliant document structure

        Example:
            >>> smells = [CodeSmell(name="Long Method", ...)]
            >>> sarif = reporter.generate_sarif(smells, Path.cwd())
            >>> print(json.dumps(sarif, indent=2))
        """
        # Build unique rules from all smells
        rules = self._build_rules(smells)

        # Convert smells to SARIF results
        results = [self._smell_to_result(smell, base_path) for smell in smells]

        # Build complete SARIF document
        sarif_doc = {
            "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": self.tool_name,
                            "version": self.tool_version,
                            "informationUri": self.tool_uri,
                            "rules": rules,
                        }
                    },
                    "results": results,
                    "invocations": [
                        {
                            "executionSuccessful": True,
                            "endTimeUtc": datetime.now(UTC).isoformat(),
                        }
                    ],
                }
            ],
        }

        return sarif_doc

    def write_sarif(
        self,
        smells: list[CodeSmell],
        output_path: Path,
        base_path: Path | None = None,
        indent: int = 2,
    ) -> None:
        """Write SARIF report to file.

        Generates SARIF document and writes it to the specified path with
        pretty-printing for readability.

        Args:
            smells: List of code smells to report
            output_path: Path where SARIF file should be written
            base_path: Base directory for relative paths (optional)
            indent: JSON indentation level (default: 2, 0 for compact)

        Raises:
            IOError: If file cannot be written
            OSError: If directory does not exist

        Example:
            >>> reporter.write_sarif(smells, Path("report.sarif"), indent=2)
        """
        sarif_doc = self.generate_sarif(smells, base_path)

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write with pretty-printing
        with open(output_path, "w", encoding="utf-8") as f:
            if indent > 0:
                json.dump(sarif_doc, f, indent=indent, ensure_ascii=False)
            else:
                json.dump(sarif_doc, f, ensure_ascii=False)

    def _severity_to_level(self, severity: SmellSeverity) -> str:
        """Map SmellSeverity to SARIF level.

        SARIF defines three result levels: error, warning, note.
        We map our SmellSeverity enum to these levels.

        Args:
            severity: SmellSeverity enum value

        Returns:
            SARIF level string ("error", "warning", or "note")

        Mapping:
            ERROR -> "error" (requires immediate attention)
            WARNING -> "warning" (should be addressed)
            INFO -> "note" (informational)
        """
        from ..collectors.smells import SmellSeverity

        mapping = {
            SmellSeverity.ERROR: "error",
            SmellSeverity.WARNING: "warning",
            SmellSeverity.INFO: "note",
        }
        return mapping.get(severity, "warning")

    def _smell_to_rule_id(self, smell_name: str) -> str:
        """Convert smell name to SARIF rule ID (kebab-case).

        Transforms human-readable smell names to kebab-case IDs suitable
        for use as SARIF rule identifiers.

        Args:
            smell_name: Human-readable smell name (e.g., "Long Method")

        Returns:
            Kebab-case rule ID (e.g., "long-method")

        Examples:
            >>> reporter._smell_to_rule_id("Long Method")
            'long-method'
            >>> reporter._smell_to_rule_id("God Class")
            'god-class'
            >>> reporter._smell_to_rule_id("Deep_Nesting")
            'deep-nesting'
        """
        return smell_name.lower().replace(" ", "-").replace("_", "-")

    def _build_rules(self, smells: list[CodeSmell]) -> list[dict[str, Any]]:
        """Generate unique rules from code smells.

        Creates SARIF rule definitions for all unique smell types found.
        Each rule includes ID, description, and help text.

        Args:
            smells: List of code smells to extract rules from

        Returns:
            List of SARIF rule objects with metadata

        SARIF Rule Structure:
            - id: Unique rule identifier (kebab-case)
            - shortDescription: Brief rule description
            - help: Detailed help text with suggestions
            - properties: Additional metadata (optional)
        """
        # Track unique smell types
        unique_smells: dict[str, CodeSmell] = {}
        for smell in smells:
            rule_id = self._smell_to_rule_id(smell.name)
            if rule_id not in unique_smells:
                unique_smells[rule_id] = smell

        # Build rule definitions
        rules = []
        for rule_id, smell in unique_smells.items():
            rule = {
                "id": rule_id,
                "shortDescription": {"text": smell.name},
                "fullDescription": {"text": self._get_smell_description(smell.name)},
            }

            # Add help text if enabled
            if self.include_help_text and smell.suggestion:
                rule["help"] = {"text": smell.suggestion}

            # Add default severity configuration
            rule["defaultConfiguration"] = {
                "level": self._severity_to_level(smell.severity)
            }

            rules.append(rule)

        return rules

    def _get_smell_description(self, smell_name: str) -> str:
        """Get detailed description for code smell type.

        Provides comprehensive descriptions for each smell type to help
        developers understand what the issue is and why it matters.

        Args:
            smell_name: Name of the code smell

        Returns:
            Detailed description explaining the smell and its impact
        """
        descriptions = {
            "Long Method": "Method or function exceeds recommended length thresholds, making it harder to understand, test, and maintain. Long methods often indicate that the function is doing too much and violates the Single Responsibility Principle.",
            "Deep Nesting": "Code has excessive nesting depth (nested if/for/while blocks), reducing readability and increasing cognitive complexity. Deep nesting makes it harder to understand control flow and increases the likelihood of bugs.",
            "Long Parameter List": "Function or method has too many parameters, making the API difficult to use and understand. Consider using parameter objects, builder pattern, or decomposing the function into smaller pieces.",
            "God Class": "Class has too many responsibilities, indicated by high method count and large size. This violates the Single Responsibility Principle and makes the class difficult to maintain, test, and reason about.",
            "Complex Method": "Method has high cyclomatic complexity (many decision points), making it difficult to test and prone to bugs. High complexity indicates complex control flow that should be simplified or decomposed.",
        }

        return descriptions.get(
            smell_name,
            f"Code smell detected: {smell_name}. Consider refactoring to improve maintainability.",
        )

    def _smell_to_result(
        self, smell: CodeSmell, base_path: Path | None = None
    ) -> dict[str, Any]:
        """Convert CodeSmell to SARIF result object.

        Transforms a CodeSmell into SARIF result format with location,
        message, and optional fingerprint for deduplication.

        Args:
            smell: Code smell to convert
            base_path: Base path for making file paths relative (optional)

        Returns:
            SARIF result object with location and message

        SARIF Result Structure:
            - ruleId: Reference to rule definition
            - level: Severity level (error/warning/note)
            - message: Human-readable message
            - locations: Where the issue was found
            - partialFingerprints: For result deduplication (optional)
        """
        # Parse location string (format: "file:line" or "file")
        location_parts = smell.location.rsplit(":", 1)
        file_path = location_parts[0]
        line_number = int(location_parts[1]) if len(location_parts) > 1 else None

        # Make path relative if base_path provided
        if base_path:
            try:
                file_path_obj = Path(file_path)
                if file_path_obj.is_absolute():
                    file_path = str(file_path_obj.relative_to(base_path))
            except (ValueError, OSError):
                # Keep original path if relative_to fails
                pass

        # Build SARIF result
        result: dict[str, Any] = {
            "ruleId": self._smell_to_rule_id(smell.name),
            "level": self._severity_to_level(smell.severity),
            "message": {
                "text": (
                    f"{smell.name}: {smell.description}"
                    if smell.description
                    else smell.name
                )
            },
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": file_path,
                        }
                    }
                }
            ],
        }

        # Add region (line number) if available
        if line_number is not None:
            result["locations"][0]["physicalLocation"]["region"] = {
                "startLine": line_number
            }

        # Add fingerprint for deduplication if enabled
        if self.include_fingerprints:
            result["partialFingerprints"] = {
                "primaryLocationLineHash": self._compute_fingerprint(smell)
            }

        # Add additional properties
        result["properties"] = {
            "metricValue": smell.metric_value,
            "threshold": smell.threshold,
        }

        return result

    def _compute_fingerprint(self, smell: CodeSmell) -> str:
        """Generate stable fingerprint for result deduplication.

        Creates a SHA-256 hash of key smell attributes to enable
        deduplication across runs and comparison of results over time.

        Args:
            smell: Code smell to fingerprint

        Returns:
            16-character hex string fingerprint

        Fingerprint Includes:
            - Smell name (type of issue)
            - Location (file:line)
            - Metric value (normalized to string)

        Example:
            >>> smell = CodeSmell(name="Long Method", location="file.py:10", ...)
            >>> fingerprint = reporter._compute_fingerprint(smell)
            >>> len(fingerprint)
            16
        """
        # Normalize metric value to avoid floating point differences
        normalized_metric = f"{smell.metric_value:.1f}"

        # Build fingerprint content
        content = f"{smell.name}:{smell.location}:{normalized_metric}"

        # Compute SHA-256 and take first 16 characters
        return hashlib.sha256(content.encode()).hexdigest()[:16]
