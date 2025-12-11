# SARIF Output Format Requirements - Issue #15 Research Report

**Date:** December 11, 2025
**Issue:** [#15 - Add SARIF Output Format](https://github.com/bobmatnyc/mcp-vector-search/issues/15)
**Milestone:** v0.18.0 - Quality Gates
**Status:** Requirements Research Complete

---

## Executive Summary

This research report provides comprehensive requirements and implementation recommendations for adding SARIF (Static Analysis Results Interchange Format) 2.1.0 output to the mcp-vector-search code analysis tool. The implementation will enable seamless GitHub Code Scanning integration and CI/CD pipeline integration for automated code quality gates.

**Key Findings:**
- SARIF 2.1.0 is the current standard with well-defined schema and GitHub integration
- Existing `CodeSmell` dataclass maps naturally to SARIF result objects
- ConsoleReporter pattern provides clear implementation template
- No external Python libraries required - native JSON generation recommended
- Implementation estimated at ~300-400 lines of code

---

## 1. SARIF 2.1.0 Format Requirements

### 1.1 Core Structure

A valid SARIF document consists of three layers:

```json
{
  "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
  "version": "2.1.0",
  "runs": [
    {
      "tool": { ... },
      "results": [ ... ]
    }
  ]
}
```

### 1.2 Required Fields

**Root Level (`sarifLog`):**
- `version`: Must be `"2.1.0"`
- `$schema`: Reference to SARIF schema URL
- `runs`: Array of run objects (at least one)

**Run Level:**
- `tool`: Tool identification and metadata
- `tool.driver`: Primary component with rules
- `tool.driver.name`: Tool name (e.g., "MCP Vector Search")
- `tool.driver.version`: Tool version from pyproject.toml
- `tool.driver.informationUri`: Project URL

**Result Level:**
- `ruleId`: Unique identifier for the rule/smell type
- `level`: Severity (`"error"`, `"warning"`, `"note"`)
- `message.text`: Human-readable description
- `locations[0].physicalLocation.artifactLocation.uri`: File path (relative to repo root)
- `locations[0].physicalLocation.region.startLine`: Line number

### 1.3 Optional But Recommended Fields

**For GitHub Code Scanning:**
- `partialFingerprints.primaryLocationLineHash`: Deduplication across runs
- `properties.security-severity`: Security severity score (0.1-10.0)
- `properties.precision`: Result precision level
- `rule.shortDescription.text`: Brief rule description
- `rule.help.markdown`: Detailed help with remediation guidance
- `rule.help.text`: Plain text help fallback

---

## 2. GitHub Code Scanning Integration

### 2.1 GitHub-Specific Requirements

**File Size Limits:**
- Maximum 10 MB (gzipped)
- Maximum 25,000 results per run
- File paths must be consistent across runs for fingerprinting

**Severity Mapping:**
- GitHub uses `level` field: `"error"`, `"warning"`, `"note"`
- Security tools should include `properties.security-severity` (0.1-10.0 scale)
- Higher severity = higher priority in UI

**Deduplication:**
- Include `partialFingerprints.primaryLocationLineHash` to prevent duplicate alerts
- GitHub matches results across runs using fingerprints
- Formula: `hash(file_path + line_number + rule_id)`

### 2.2 Best Practices for GitHub

1. **Relative File Paths**: Use paths relative to repository root
2. **Consistent Paths**: Ensure same file always has same path across runs
3. **Rich Descriptions**: Provide both `text` and `markdown` help
4. **Rule Metadata**: Include complete rule definitions in `tool.driver.rules`
5. **Line Numbers**: Always include `startLine`, optionally `endLine`
6. **Code Regions**: Optionally include `snippet.text` for context

---

## 3. Mapping Code Smells to SARIF

### 3.1 CodeSmell → SARIF Result Mapping

| CodeSmell Field | SARIF Field | Mapping Notes |
|----------------|-------------|---------------|
| `name` | `ruleId` | Normalize to kebab-case (e.g., "long-method") |
| `name` | `message.text` | Use as primary message component |
| `description` | `message.text` | Append to message for full context |
| `severity` | `level` | Map: ERROR→error, WARNING→warning, INFO→note |
| `location` | `physicalLocation.artifactLocation.uri` | Parse file path from "file:line" format |
| `location` | `physicalLocation.region.startLine` | Parse line number from location string |
| `metric_value` | `message.text` | Include in message: "value: X, threshold: Y" |
| `threshold` | `message.text` | Include in message for context |
| `suggestion` | `fixes[0].description.text` | Map to remediation fix (optional) |

### 3.2 Severity Mapping

```python
# SmellSeverity → SARIF level
SEVERITY_MAP = {
    SmellSeverity.ERROR: "error",
    SmellSeverity.WARNING: "warning",
    SmellSeverity.INFO: "note",
}
```

### 3.3 Rule ID Generation

```python
# Convert smell names to rule IDs
def smell_to_rule_id(smell_name: str) -> str:
    """
    Convert smell name to SARIF rule ID.

    Examples:
        "Long Method" → "long-method"
        "Deep Nesting" → "deep-nesting"
        "God Class" → "god-class"
    """
    return smell_name.lower().replace(" ", "-")
```

### 3.4 Location Parsing

Current format: `"src/example.py:42"` or `"src/example.py:10-50"`

```python
def parse_location(location: str) -> tuple[str, int, int | None]:
    """
    Parse location string to components.

    Args:
        location: Format "file:line" or "file:start-end"

    Returns:
        (file_path, start_line, end_line or None)
    """
    if ":" not in location:
        return location, 1, None

    file_path, line_part = location.rsplit(":", 1)

    if "-" in line_part:
        start, end = line_part.split("-")
        return file_path, int(start), int(end)
    else:
        return file_path, int(line_part), None
```

---

## 4. Implementation Recommendations

### 4.1 File Structure

```
src/mcp_vector_search/analysis/reporters/
├── __init__.py          # Export SARIFReporter
├── console.py           # Existing ConsoleReporter
└── sarif.py             # New SARIFReporter (THIS ISSUE)
```

### 4.2 SARIFReporter Class Design

```python
"""SARIF 2.1.0 reporter for code analysis results."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..collectors.smells import CodeSmell
    from ..metrics import ProjectMetrics

class SARIFReporter:
    """Reporter for generating SARIF 2.1.0 format output."""

    def __init__(
        self,
        tool_name: str = "MCP Vector Search",
        tool_version: str = "0.18.0",
        tool_uri: str = "https://github.com/bobmatnyc/mcp-vector-search",
    ) -> None:
        """Initialize SARIF reporter."""
        self.tool_name = tool_name
        self.tool_version = tool_version
        self.tool_uri = tool_uri

    def generate_sarif(
        self,
        smells: list[CodeSmell],
        project_root: Path | None = None,
    ) -> dict[str, Any]:
        """
        Generate SARIF 2.1.0 document from code smells.

        Args:
            smells: List of detected code smells
            project_root: Project root for relative paths (optional)

        Returns:
            SARIF document as dictionary
        """
        # Build rules from unique smell types
        rules = self._build_rules(smells)

        # Convert smells to SARIF results
        results = [
            self._smell_to_result(smell, project_root)
            for smell in smells
        ]

        return {
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
                }
            ],
        }

    def write_sarif(
        self,
        smells: list[CodeSmell],
        output_path: Path,
        project_root: Path | None = None,
    ) -> None:
        """
        Write SARIF output to file.

        Args:
            smells: List of detected code smells
            output_path: Path to write SARIF file
            project_root: Project root for relative paths
        """
        sarif_doc = self.generate_sarif(smells, project_root)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sarif_doc, f, indent=2, ensure_ascii=False)

    def _build_rules(self, smells: list[CodeSmell]) -> list[dict[str, Any]]:
        """Build SARIF rules from unique smell types."""
        # Get unique smell names
        unique_smells = {smell.name for smell in smells}

        rules = []
        for smell_name in sorted(unique_smells):
            rule_id = self._smell_to_rule_id(smell_name)

            # Get representative smell for this type
            representative = next(s for s in smells if s.name == smell_name)

            rules.append({
                "id": rule_id,
                "shortDescription": {
                    "text": smell_name
                },
                "fullDescription": {
                    "text": representative.description
                },
                "help": {
                    "text": representative.suggestion or f"Refactor to address {smell_name}",
                    "markdown": self._generate_help_markdown(smell_name, representative),
                },
                "defaultConfiguration": {
                    "level": self._severity_to_level(representative.severity),
                },
            })

        return rules

    def _smell_to_result(
        self,
        smell: CodeSmell,
        project_root: Path | None = None,
    ) -> dict[str, Any]:
        """Convert CodeSmell to SARIF result."""
        file_path, start_line, end_line = self._parse_location(smell.location)

        # Make path relative if project_root provided
        if project_root:
            try:
                file_path = str(Path(file_path).relative_to(project_root))
            except ValueError:
                pass  # Keep absolute path if not under project_root

        result: dict[str, Any] = {
            "ruleId": self._smell_to_rule_id(smell.name),
            "level": self._severity_to_level(smell.severity),
            "message": {
                "text": self._format_message(smell)
            },
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": file_path,
                        },
                        "region": {
                            "startLine": start_line,
                        },
                    }
                }
            ],
        }

        # Add endLine if available
        if end_line:
            result["locations"][0]["physicalLocation"]["region"]["endLine"] = end_line

        # Add fingerprint for GitHub deduplication
        result["partialFingerprints"] = {
            "primaryLocationLineHash": self._compute_fingerprint(
                file_path, start_line, self._smell_to_rule_id(smell.name)
            )
        }

        return result

    def _smell_to_rule_id(self, smell_name: str) -> str:
        """Convert smell name to rule ID (kebab-case)."""
        return smell_name.lower().replace(" ", "-")

    def _severity_to_level(self, severity: SmellSeverity) -> str:
        """Convert SmellSeverity to SARIF level."""
        from ..collectors.smells import SmellSeverity

        mapping = {
            SmellSeverity.ERROR: "error",
            SmellSeverity.WARNING: "warning",
            SmellSeverity.INFO: "note",
        }
        return mapping[severity]

    def _parse_location(self, location: str) -> tuple[str, int, int | None]:
        """Parse location string to components."""
        if ":" not in location:
            return location, 1, None

        file_path, line_part = location.rsplit(":", 1)

        if "-" in line_part:
            start, end = line_part.split("-")
            return file_path, int(start), int(end)
        else:
            return file_path, int(line_part), None

    def _format_message(self, smell: CodeSmell) -> str:
        """Format smell as SARIF message."""
        return (
            f"{smell.name}: {smell.description} "
            f"(value: {smell.metric_value}, threshold: {smell.threshold})"
        )

    def _compute_fingerprint(
        self,
        file_path: str,
        line: int,
        rule_id: str,
    ) -> str:
        """Compute fingerprint for result deduplication."""
        import hashlib

        # GitHub-compatible fingerprint
        fingerprint_input = f"{file_path}:{line}:{rule_id}"
        return hashlib.sha256(fingerprint_input.encode()).hexdigest()[:16]

    def _generate_help_markdown(
        self,
        smell_name: str,
        representative: CodeSmell,
    ) -> str:
        """Generate markdown help for rule."""
        return f"""## {smell_name}

{representative.description}

### Recommendation

{representative.suggestion or f"Refactor code to address {smell_name}."}

### References

- [Code Smell: {smell_name}](https://refactoring.guru/smells/{self._smell_to_rule_id(smell_name)})
"""
```

### 4.3 CLI Integration

**Update `src/mcp_vector_search/cli/commands/analyze.py`:**

```python
# Add new option
@analyze_app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    # ... existing options ...
    output_format: str = typer.Option(
        "console",
        "--format",
        "-f",
        help="Output format: console, json, sarif",
        rich_help_panel="📊 Display Options",
    ),
    output_file: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write output to file (required for SARIF format)",
        rich_help_panel="📊 Display Options",
    ),
) -> None:
    """Analyze code complexity and quality."""
    # ... existing code ...

    # Add SARIF output handling
    if output_format == "sarif":
        if not output_file:
            print_error("--output/-o required when using --format sarif")
            raise typer.Exit(1)

        from ...analysis.reporters.sarif import SARIFReporter

        reporter = SARIFReporter(
            tool_version=get_tool_version(),  # From __version__
        )
        reporter.write_sarif(
            smells=all_smells,
            output_path=output_file,
            project_root=project_root,
        )
        print_info(f"SARIF output written to: {output_file}")
```

### 4.4 Version Extraction

**Create utility in `src/mcp_vector_search/utils/version.py`:**

```python
"""Version utilities."""

from importlib.metadata import version

def get_tool_version() -> str:
    """Get tool version from package metadata."""
    try:
        return version("mcp-vector-search")
    except Exception:
        return "0.0.0-dev"
```

---

## 5. Testing Strategy

### 5.1 Unit Tests

**File: `tests/unit/analysis/reporters/test_sarif.py`**

```python
"""Tests for SARIF reporter."""

import json
from pathlib import Path

import pytest

from mcp_vector_search.analysis.collectors.smells import CodeSmell, SmellSeverity
from mcp_vector_search.analysis.reporters.sarif import SARIFReporter


def test_generate_sarif_minimal():
    """Test minimal SARIF generation."""
    reporter = SARIFReporter()

    smells = [
        CodeSmell(
            name="Long Method",
            description="Method is too long: 75 lines (threshold: 50)",
            severity=SmellSeverity.WARNING,
            location="src/example.py:10",
            metric_value=75.0,
            threshold=50.0,
            suggestion="Break into smaller functions",
        )
    ]

    sarif = reporter.generate_sarif(smells)

    # Validate structure
    assert sarif["version"] == "2.1.0"
    assert "$schema" in sarif
    assert len(sarif["runs"]) == 1

    run = sarif["runs"][0]
    assert run["tool"]["driver"]["name"] == "MCP Vector Search"
    assert len(run["results"]) == 1

    result = run["results"][0]
    assert result["ruleId"] == "long-method"
    assert result["level"] == "warning"
    assert result["locations"][0]["physicalLocation"]["region"]["startLine"] == 10


def test_severity_mapping():
    """Test severity to SARIF level mapping."""
    reporter = SARIFReporter()

    assert reporter._severity_to_level(SmellSeverity.ERROR) == "error"
    assert reporter._severity_to_level(SmellSeverity.WARNING) == "warning"
    assert reporter._severity_to_level(SmellSeverity.INFO) == "note"


def test_location_parsing():
    """Test location string parsing."""
    reporter = SARIFReporter()

    # Single line
    file, start, end = reporter._parse_location("src/example.py:42")
    assert file == "src/example.py"
    assert start == 42
    assert end is None

    # Range
    file, start, end = reporter._parse_location("src/example.py:10-50")
    assert file == "src/example.py"
    assert start == 10
    assert end == 50


def test_rule_generation():
    """Test SARIF rule generation from smells."""
    reporter = SARIFReporter()

    smells = [
        CodeSmell(
            name="Long Method",
            description="Test",
            severity=SmellSeverity.WARNING,
            location="a.py:1",
            metric_value=1.0,
            threshold=1.0,
        ),
        CodeSmell(
            name="Long Method",
            description="Test",
            severity=SmellSeverity.WARNING,
            location="b.py:1",
            metric_value=1.0,
            threshold=1.0,
        ),
        CodeSmell(
            name="Deep Nesting",
            description="Test",
            severity=SmellSeverity.WARNING,
            location="c.py:1",
            metric_value=1.0,
            threshold=1.0,
        ),
    ]

    rules = reporter._build_rules(smells)

    # Should have 2 unique rules
    assert len(rules) == 2
    rule_ids = {rule["id"] for rule in rules}
    assert rule_ids == {"long-method", "deep-nesting"}


def test_relative_path_conversion():
    """Test relative path conversion."""
    reporter = SARIFReporter()
    project_root = Path("/home/user/project")

    smell = CodeSmell(
        name="Long Method",
        description="Test",
        severity=SmellSeverity.WARNING,
        location="/home/user/project/src/example.py:10",
        metric_value=1.0,
        threshold=1.0,
    )

    result = reporter._smell_to_result(smell, project_root)
    uri = result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]

    assert uri == "src/example.py"


def test_write_sarif_file(tmp_path):
    """Test writing SARIF to file."""
    reporter = SARIFReporter()
    output_file = tmp_path / "output.sarif"

    smells = [
        CodeSmell(
            name="Long Method",
            description="Test",
            severity=SmellSeverity.WARNING,
            location="src/example.py:10",
            metric_value=1.0,
            threshold=1.0,
        )
    ]

    reporter.write_sarif(smells, output_file)

    assert output_file.exists()

    # Validate JSON
    with open(output_file) as f:
        sarif = json.load(f)

    assert sarif["version"] == "2.1.0"
```

### 5.2 Integration Tests

**File: `tests/integration/test_sarif_integration.py`**

```python
"""Integration tests for SARIF output."""

import json
from pathlib import Path

import pytest


def test_analyze_sarif_output(tmp_path, sample_project):
    """Test analyze command with SARIF output."""
    output_file = tmp_path / "results.sarif"

    # Run analysis with SARIF output
    result = subprocess.run(
        [
            "mcp-vector-search",
            "analyze",
            "--format", "sarif",
            "--output", str(output_file),
            "--project-root", str(sample_project),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert output_file.exists()

    # Validate SARIF structure
    with open(output_file) as f:
        sarif = json.load(f)

    assert sarif["version"] == "2.1.0"
    assert len(sarif["runs"]) == 1


def test_sarif_schema_validation(tmp_path, sample_project):
    """Test SARIF output validates against schema."""
    output_file = tmp_path / "results.sarif"

    # Generate SARIF
    # ... run analysis ...

    # Download and validate against schema
    schema_url = "https://json.schemastore.org/sarif-2.1.0.json"
    # Use jsonschema library for validation
    import jsonschema
    import requests

    schema = requests.get(schema_url).json()

    with open(output_file) as f:
        sarif = json.load(f)

    # Should not raise ValidationError
    jsonschema.validate(sarif, schema)
```

### 5.3 GitHub Code Scanning Test

**Manual Test Procedure:**

1. Generate SARIF file for test project:
   ```bash
   mcp-vector-search analyze --format sarif --output results.sarif
   ```

2. Upload to GitHub via API:
   ```bash
   gh api \
     --method POST \
     -H "Accept: application/vnd.github+json" \
     repos/OWNER/REPO/code-scanning/sarifs \
     -F "sarif=@results.sarif" \
     -F "commit_sha=$GITHUB_SHA" \
     -F "ref=refs/heads/main"
   ```

3. Verify in GitHub UI:
   - Navigate to Security → Code Scanning
   - Verify alerts appear for detected smells
   - Check severity mapping is correct
   - Verify file locations link to correct code

---

## 6. Example SARIF Output

### 6.1 Complete Example

```json
{
  "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
  "version": "2.1.0",
  "runs": [
    {
      "tool": {
        "driver": {
          "name": "MCP Vector Search",
          "version": "0.18.0",
          "informationUri": "https://github.com/bobmatnyc/mcp-vector-search",
          "rules": [
            {
              "id": "long-method",
              "shortDescription": {
                "text": "Long Method"
              },
              "fullDescription": {
                "text": "Method/function is too long: 75 lines (threshold: 50)"
              },
              "help": {
                "text": "Consider breaking this method into smaller, focused functions",
                "markdown": "## Long Method\n\nMethod/function is too long: 75 lines (threshold: 50)\n\n### Recommendation\n\nConsider breaking this method into smaller, focused functions\n\n### References\n\n- [Code Smell: Long Method](https://refactoring.guru/smells/long-method)\n"
              },
              "defaultConfiguration": {
                "level": "warning"
              }
            },
            {
              "id": "deep-nesting",
              "shortDescription": {
                "text": "Deep Nesting"
              },
              "fullDescription": {
                "text": "Excessive nesting depth: 6 levels (threshold: 4)"
              },
              "help": {
                "text": "Consider extracting nested logic into separate functions or using early returns",
                "markdown": "## Deep Nesting\n\nExcessive nesting depth: 6 levels (threshold: 4)\n\n### Recommendation\n\nConsider extracting nested logic into separate functions or using early returns\n\n### References\n\n- [Code Smell: Deep Nesting](https://refactoring.guru/smells/deep-nesting)\n"
              },
              "defaultConfiguration": {
                "level": "warning"
              }
            },
            {
              "id": "god-class",
              "shortDescription": {
                "text": "God Class"
              },
              "fullDescription": {
                "text": "Class has too many responsibilities: 25 methods (threshold: 20), 650 lines (threshold: 500)"
              },
              "help": {
                "text": "Split this class into smaller, focused classes following Single Responsibility Principle",
                "markdown": "## God Class\n\nClass has too many responsibilities: 25 methods (threshold: 20), 650 lines (threshold: 500)\n\n### Recommendation\n\nSplit this class into smaller, focused classes following Single Responsibility Principle\n\n### References\n\n- [Code Smell: God Class](https://refactoring.guru/smells/god-class)\n"
              },
              "defaultConfiguration": {
                "level": "error"
              }
            }
          ]
        }
      },
      "results": [
        {
          "ruleId": "long-method",
          "level": "warning",
          "message": {
            "text": "Long Method: Method/function is too long: 75 lines (threshold: 50) (value: 75.0, threshold: 50.0)"
          },
          "locations": [
            {
              "physicalLocation": {
                "artifactLocation": {
                  "uri": "src/mcp_vector_search/core/indexer.py"
                },
                "region": {
                  "startLine": 145
                }
              }
            }
          ],
          "partialFingerprints": {
            "primaryLocationLineHash": "a3f5b8c9d2e1f4a6"
          }
        },
        {
          "ruleId": "deep-nesting",
          "level": "warning",
          "message": {
            "text": "Deep Nesting: Excessive nesting depth: 6 levels (threshold: 4) (value: 6.0, threshold: 4.0)"
          },
          "locations": [
            {
              "physicalLocation": {
                "artifactLocation": {
                  "uri": "src/mcp_vector_search/core/search.py"
                },
                "region": {
                  "startLine": 89
                }
              }
            }
          ],
          "partialFingerprints": {
            "primaryLocationLineHash": "b7c4d9e2f3a1b5c8"
          }
        },
        {
          "ruleId": "god-class",
          "level": "error",
          "message": {
            "text": "God Class: Class has too many responsibilities: 25 methods (threshold: 20), 650 lines (threshold: 500) (value: 25.0, threshold: 20.0)"
          },
          "locations": [
            {
              "physicalLocation": {
                "artifactLocation": {
                  "uri": "src/mcp_vector_search/parsers/base.py"
                },
                "region": {
                  "startLine": 1
                }
              }
            }
          ],
          "partialFingerprints": {
            "primaryLocationLineHash": "c9d8e7f6a5b4c3d2"
          }
        }
      ]
    }
  ]
}
```

---

## 7. Implementation Checklist

### Phase 1: Core Implementation
- [ ] Create `src/mcp_vector_search/analysis/reporters/sarif.py`
- [ ] Implement `SARIFReporter` class with core methods
- [ ] Implement severity mapping (ERROR→error, WARNING→warning, INFO→note)
- [ ] Implement location parsing from "file:line" format
- [ ] Implement rule generation from unique smell types
- [ ] Implement result generation from CodeSmell objects
- [ ] Implement relative path conversion
- [ ] Implement fingerprint generation for deduplication

### Phase 2: CLI Integration
- [ ] Add `--format sarif` option to analyze command
- [ ] Add `--output` option for SARIF file path
- [ ] Implement version extraction utility
- [ ] Update `__init__.py` to export SARIFReporter
- [ ] Add validation: require `--output` when format is SARIF

### Phase 3: Testing
- [ ] Write unit tests for SARIFReporter class
- [ ] Write unit tests for severity mapping
- [ ] Write unit tests for location parsing
- [ ] Write unit tests for rule generation
- [ ] Write integration tests for CLI SARIF output
- [ ] Write schema validation tests
- [ ] Manual test: Upload to GitHub Code Scanning
- [ ] Manual test: Verify alerts appear in PR diffs

### Phase 4: Documentation
- [ ] Update README.md with SARIF output examples
- [ ] Create `docs/guides/sarif-integration.md`
- [ ] Add SARIF examples to `examples/` directory
- [ ] Update CHANGELOG.md with SARIF feature
- [ ] Add docstrings to all SARIFReporter methods
- [ ] Create CI/CD integration guide

---

## 8. Dependencies and Blockers

### 8.1 Blocked By
- **Issue #10**: _(Check status - may need code smell detection to be complete)_
- **Issue #14**: _(Check status - may need certain metrics to be available)_

### 8.2 Blocks
- **Issue #16**: _(SARIF output may be prerequisite for CI integration)_

### 8.3 Related
- **Epic #2**: Quality Gates - Phase 2

### 8.4 Python Dependencies
**No new dependencies required!** Use built-in libraries:
- `json` - SARIF generation
- `hashlib` - Fingerprint generation
- `pathlib` - Path manipulation
- `importlib.metadata` - Version extraction

---

## 9. Risk Analysis

### 9.1 Implementation Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| SARIF schema validation failures | Medium | High | Thorough testing with schema validator |
| GitHub Code Scanning integration issues | Low | High | Follow GitHub documentation precisely |
| Large file size (>10MB limit) | Low | Medium | Implement result truncation if needed |
| Path resolution issues (absolute vs relative) | Medium | Medium | Robust path normalization logic |
| Fingerprint collision | Very Low | Low | Use SHA-256 hash (low collision probability) |

### 9.2 Mitigation Strategies

1. **Schema Validation**: Use `jsonschema` library in tests to validate output
2. **GitHub Testing**: Set up test repository for SARIF uploads before production
3. **File Size Monitoring**: Log warning if approaching 10MB limit, implement truncation
4. **Path Testing**: Test with various project structures (nested, flat, symlinks)
5. **Fingerprint Uniqueness**: Include file path + line + rule ID in hash input

---

## 10. Future Enhancements

### 10.1 Potential Improvements (Post-v0.18.0)

1. **Code Snippets**: Include `snippet.text` in region for context
2. **Fix Suggestions**: Add `fixes` array with concrete refactoring suggestions
3. **Taxonomy Mapping**: Map smells to CWE/OWASP categories for security tools
4. **Streaming Output**: Generate SARIF incrementally for large projects
5. **Compressed Output**: Support gzipped SARIF for large files
6. **Multiple Tools**: Support multiple "tool" runs in single SARIF file
7. **Thread Flow**: Add execution flow information for complex smells
8. **Graph Support**: Leverage existing D3.js visualization for smell relationships

### 10.2 Integration Opportunities

1. **GitHub Actions**: Create action for automatic SARIF upload
2. **GitLab CI**: Support GitLab Code Quality format
3. **Azure DevOps**: Support Azure Pipelines integration
4. **SonarQube**: Convert SARIF to SonarQube format
5. **IDE Integration**: VSCode extension to visualize SARIF results

---

## 11. References

### 11.1 Specifications
- [SARIF 2.1.0 Specification](https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html)
- [SARIF Schema](https://json.schemastore.org/sarif-2.1.0.json)
- [GitHub Code Scanning SARIF Support](https://docs.github.com/en/code-security/code-scanning/integrating-with-code-scanning/sarif-support-for-code-scanning)

### 11.2 Python Libraries
- [microsoft/sarif-python-om](https://github.com/microsoft/sarif-python-om) - Official Python object model (not recommended - no serialization support)
- [pysarif](https://pypi.org/project/pysarif/) - Python SARIF library with load/save (optional, not required)
- [microsoft/sarif-tools](https://github.com/microsoft/sarif-tools) - Command-line tools for SARIF (useful for testing)

### 11.3 Examples
- [SARIF Tutorials](https://github.com/microsoft/sarif-tutorials) - Microsoft SARIF examples
- [CodeQL SARIF Output](https://github.com/github/codeql-action) - GitHub's CodeQL SARIF examples

---

## 12. Conclusion

The SARIF output format implementation is **straightforward and well-specified**. The existing `CodeSmell` dataclass maps naturally to SARIF result objects, and no external dependencies are required beyond Python's standard library.

**Key Success Factors:**
1. Follow SARIF 2.1.0 specification precisely
2. Implement GitHub-specific best practices (fingerprinting, relative paths)
3. Comprehensive testing with schema validation
4. Clear CLI interface with `--format sarif` and `--output` options

**Estimated Effort:** 2-3 days for full implementation including tests and documentation.

**Next Steps:**
1. Verify Issue #10 and #14 are complete (blockers)
2. Begin implementation with Phase 1 (Core Implementation)
3. Set up test GitHub repository for SARIF upload testing
4. Create PR following standard workflow

---

**Research Completed By:** Claude Code (Research Agent)
**Date:** December 11, 2025
**Issue:** #15 - Add SARIF Output Format
