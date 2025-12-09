#!/usr/bin/env python3
"""
Homebrew Formula Update Automation Script

Automates the process of updating the Homebrew formula for mcp-vector-search:
1. Fetches latest version and SHA256 from PyPI
2. Clones/updates the homebrew-mcp-vector-search tap repository
3. Updates the Formula file with new version and hash
4. Commits and pushes changes to the tap repository

Features:
- Dry-run mode for safe testing
- Automatic rollback on failure
- Detailed logging with rich console output
- CI-friendly exit codes
- GitHub token authentication via environment variables

Usage:
    # Update to latest PyPI version (dry-run)
    ./scripts/update_homebrew_formula.py --dry-run

    # Update to latest version and push
    ./scripts/update_homebrew_formula.py

    # Update to specific version
    ./scripts/update_homebrew_formula.py --version 0.12.8

    # Specify custom tap repo path
    ./scripts/update_homebrew_formula.py --tap-repo-path /path/to/tap

Environment Variables:
    HOMEBREW_TAP_TOKEN: GitHub token for authentication (required for push)
    HOMEBREW_TAP_REPO: URL of tap repository (default: bobmatnyc/homebrew-mcp-vector-search)

Exit Codes:
    0: Success
    1: PyPI API error
    2: Git operation error
    3: Formula update error
    4: Validation error
    5: Authentication error
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Color codes for terminal output
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
BLUE = "\033[0;34m"
CYAN = "\033[0;36m"
BOLD = "\033[1m"
RESET = "\033[0m"


@dataclass
class PackageInfo:
    """Package information from PyPI."""

    version: str
    url: str
    sha256: str
    size: int


class HomebrewFormulaUpdater:
    """Manages updates to Homebrew formula for mcp-vector-search."""

    def __init__(
        self,
        tap_repo_path: Path | None = None,
        tap_repo_url: str | None = None,
        dry_run: bool = False,
        verbose: bool = False,
    ):
        """Initialize the formula updater.

        Args:
            tap_repo_path: Local path to tap repository
            tap_repo_url: URL of tap repository
            dry_run: If True, show what would be done without making changes
            verbose: Enable verbose logging
        """
        self.dry_run = dry_run
        self.verbose = verbose

        # Default tap repository
        self.tap_repo_url = tap_repo_url or os.getenv(
            "HOMEBREW_TAP_REPO",
            "https://github.com/bobmatnyc/homebrew-mcp-vector-search.git",
        )

        # Default tap repo path
        if tap_repo_path:
            self.tap_repo_path = tap_repo_path
        else:
            self.tap_repo_path = (
                Path.home() / ".homebrew_tap_update" / "homebrew-mcp-vector-search"
            )

        # GitHub token for authentication
        self.github_token = os.getenv("HOMEBREW_TAP_TOKEN")

        # Formula file path (relative to repo root)
        self.formula_name = "mcp-vector-search.rb"

        # Track changes for rollback
        self.backup_path: Path | None = None
        self.created_commit = False

    def log(self, message: str, level: str = "info") -> None:
        """Log a message with color coding.

        Args:
            message: Message to log
            level: Log level (info, success, warning, error, debug)
        """
        colors = {
            "info": BLUE,
            "success": GREEN,
            "warning": YELLOW,
            "error": RED,
            "debug": CYAN,
        }

        symbols = {
            "info": "ℹ",
            "success": "✓",
            "warning": "⚠",
            "error": "✗",
            "debug": "→",
        }

        if level == "debug" and not self.verbose:
            return

        color = colors.get(level, "")
        symbol = symbols.get(level, "•")

        if self.dry_run and level in ("info", "success"):
            prefix = f"{YELLOW}[DRY RUN]{RESET} "
        else:
            prefix = ""

        print(f"{prefix}{color}{symbol}{RESET} {message}")

    def fetch_pypi_info(self, version: str | None = None) -> PackageInfo:
        """Fetch package information from PyPI.

        Args:
            version: Specific version to fetch, or None for latest

        Returns:
            PackageInfo with version, URL, and SHA256

        Raises:
            SystemExit: If PyPI API request fails
        """
        self.log("Fetching package information from PyPI...", "info")

        # PyPI JSON API endpoint
        if version:
            url = f"https://pypi.org/pypi/mcp-vector-search/{version}/json"
        else:
            url = "https://pypi.org/pypi/mcp-vector-search/json"

        try:
            self.log(f"Requesting: {url}", "debug")

            request = Request(url)
            request.add_header("User-Agent", "mcp-vector-search-formula-updater/1.0")

            with urlopen(request, timeout=30) as response:  # nosec B310
                data = json.loads(response.read().decode())

            # Extract version
            pkg_version = data["info"]["version"]

            # Find sdist (source distribution)
            sdist = None
            for release in data["urls"]:
                if release["packagetype"] == "sdist":
                    sdist = release
                    break

            if not sdist:
                self.log("No source distribution found on PyPI", "error")
                sys.exit(1)

            # Validate SHA256
            if "digests" not in sdist or "sha256" not in sdist["digests"]:
                self.log("SHA256 hash not found in PyPI response", "error")
                sys.exit(1)

            pkg_info = PackageInfo(
                version=pkg_version,
                url=sdist["url"],
                sha256=sdist["digests"]["sha256"],
                size=sdist["size"],
            )

            self.log(f"Found version: {BOLD}{pkg_version}{RESET}", "success")
            self.log(f"URL: {pkg_info.url}", "debug")
            self.log(f"SHA256: {pkg_info.sha256}", "debug")
            self.log(f"Size: {pkg_info.size:,} bytes", "debug")

            return pkg_info

        except HTTPError as e:
            if e.code == 404:
                self.log(f"Version {version or 'latest'} not found on PyPI", "error")
            else:
                self.log(f"HTTP error fetching PyPI data: {e.code} {e.reason}", "error")
            sys.exit(1)
        except URLError as e:
            self.log(f"Network error fetching PyPI data: {e.reason}", "error")
            sys.exit(1)
        except (KeyError, json.JSONDecodeError) as e:
            self.log(f"Invalid PyPI response format: {e}", "error")
            sys.exit(1)

    def verify_sha256(self, package_info: PackageInfo) -> bool:
        """Verify SHA256 hash by downloading the package.

        Args:
            package_info: Package information to verify

        Returns:
            True if hash matches, False otherwise
        """
        self.log("Verifying SHA256 hash integrity...", "info")

        if self.dry_run:
            self.log("Skipping verification in dry-run mode", "debug")
            return True

        try:
            # Download package
            request = Request(package_info.url)
            request.add_header("User-Agent", "mcp-vector-search-formula-updater/1.0")

            with urlopen(request, timeout=60) as response:  # nosec B310
                data = response.read()

            # Calculate SHA256
            calculated_hash = hashlib.sha256(data).hexdigest()

            if calculated_hash == package_info.sha256:
                self.log("SHA256 hash verified successfully", "success")
                return True
            else:
                self.log("SHA256 mismatch!", "error")
                self.log(f"Expected: {package_info.sha256}", "error")
                self.log(f"Got: {calculated_hash}", "error")
                return False

        except Exception as e:
            self.log(f"Error verifying hash: {e}", "warning")
            return True  # Don't fail on verification error

    def setup_tap_repository(self) -> None:
        """Clone or update the tap repository.

        Raises:
            SystemExit: If git operations fail
        """
        if self.tap_repo_path.exists():
            self.log(f"Tap repository exists at {self.tap_repo_path}", "info")

            if self.dry_run:
                self.log("Would pull latest changes", "debug")
                return

            # Pull latest changes
            try:
                self.log("Pulling latest changes...", "info")
                result = subprocess.run(  # nosec B607
                    ["git", "-C", str(self.tap_repo_path), "pull"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.log(
                    result.stdout.strip() if result.stdout else "Already up to date",
                    "debug",
                )
                self.log("Repository updated", "success")
            except subprocess.CalledProcessError as e:
                self.log(f"Failed to pull changes: {e.stderr}", "error")
                sys.exit(2)
        else:
            self.log(f"Cloning tap repository to {self.tap_repo_path}", "info")

            if self.dry_run:
                self.log(f"Would clone from {self.tap_repo_url}", "debug")
                return

            # Clone repository
            try:
                self.tap_repo_path.parent.mkdir(parents=True, exist_ok=True)

                result = subprocess.run(  # nosec B607
                    ["git", "clone", self.tap_repo_url, str(self.tap_repo_path)],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.log("Repository cloned successfully", "success")
            except subprocess.CalledProcessError as e:
                self.log(f"Failed to clone repository: {e.stderr}", "error")
                sys.exit(2)

    def backup_formula(self, formula_path: Path) -> None:
        """Create a backup of the current formula.

        Args:
            formula_path: Path to formula file
        """
        if self.dry_run:
            return

        backup_name = (
            f"{formula_path.name}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.backup_path = formula_path.parent / backup_name

        shutil.copy2(formula_path, self.backup_path)
        self.log(f"Created backup: {self.backup_path.name}", "debug")

    def update_formula(self, package_info: PackageInfo) -> Path:
        """Update the Formula file with new version and hash.

        Args:
            package_info: Package information to update

        Returns:
            Path to updated formula file

        Raises:
            SystemExit: If formula update fails
        """
        formula_path = self.tap_repo_path / "Formula" / self.formula_name

        # Check if old location exists (root directory)
        old_formula_path = self.tap_repo_path / self.formula_name
        if not formula_path.exists() and old_formula_path.exists():
            formula_path = old_formula_path
            self.log("Using formula from root directory (old structure)", "debug")

        if not formula_path.exists():
            self.log(f"Formula file not found: {formula_path}", "error")
            sys.exit(3)

        self.log(f"Updating formula: {formula_path.name}", "info")

        # Backup original
        self.backup_formula(formula_path)

        # Read current formula
        try:
            content = formula_path.read_text()
        except Exception as e:
            self.log(f"Failed to read formula: {e}", "error")
            sys.exit(3)

        # Extract current version and sha256
        # First try explicit version directive
        version_match = re.search(r'version\s+"([^"]+)"', content)
        if version_match:
            current_version = version_match.group(1)
        else:
            # Extract version from URL (e.g., mcp_vector_search-0.15.16.tar.gz)
            url_match = re.search(
                r"mcp_vector_search-([0-9]+\.[0-9]+\.[0-9]+)\.tar\.gz", content
            )
            if url_match:
                current_version = url_match.group(1)
            else:
                self.log(
                    "Could not parse version from formula (tried version directive and URL)",
                    "error",
                )
                sys.exit(3)

        sha256_match = re.search(r'sha256\s+"([^"]+)"', content)
        if not sha256_match:
            self.log("Could not parse sha256 from formula", "error")
            sys.exit(3)

        current_sha256 = sha256_match.group(1)

        if current_version == package_info.version:
            self.log(f"Formula already at version {current_version}", "warning")
            if current_sha256 == package_info.sha256:
                self.log("No changes needed", "info")
                return formula_path

        self.log(
            f"Version: {current_version} → {BOLD}{package_info.version}{RESET}", "info"
        )

        # Show diff
        if self.verbose:
            self.log("Changes:", "debug")
            self.log(f'  - version "{current_version}"', "debug")
            self.log(f'  + version "{package_info.version}"', "debug")
            self.log(f'  - sha256 "{current_sha256}"', "debug")
            self.log(f'  + sha256 "{package_info.sha256}"', "debug")

        if self.dry_run:
            self.log("Would update formula file", "debug")
            return formula_path

        # Update version directive only if it exists
        if version_match:
            content = re.sub(
                r'version\s+"[^"]+"', f'version "{package_info.version}"', content
            )

        # Update sha256
        content = re.sub(
            r'sha256\s+"[^"]+"', f'sha256 "{package_info.sha256}"', content
        )

        # Update URL (this also updates version implicitly for URL-derived versions)
        content = re.sub(
            r'url\s+"https://files\.pythonhosted\.org/packages/[^"]+/mcp[_-]vector[_-]search-[^"]+\.tar\.gz"',
            f'url "{package_info.url}"',
            content,
        )

        # Write updated formula
        try:
            formula_path.write_text(content)
            self.log("Formula file updated", "success")
        except Exception as e:
            self.log(f"Failed to write formula: {e}", "error")
            self.rollback()
            sys.exit(3)

        return formula_path

    def validate_formula(self, formula_path: Path) -> bool:
        """Validate Ruby syntax of formula (optional).

        Args:
            formula_path: Path to formula file

        Returns:
            True if valid, False otherwise
        """
        if self.dry_run:
            return True

        self.log("Validating formula syntax...", "info")

        # Check if ruby is available
        try:
            result = subprocess.run(  # nosec B607
                ["ruby", "-c", str(formula_path)],
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode == 0:
                self.log("Formula syntax valid", "success")
                return True
            else:
                self.log(f"Formula syntax error: {result.stderr}", "error")
                return False

        except FileNotFoundError:
            self.log("Ruby not found, skipping syntax validation", "warning")
            return True

    def commit_and_push(self, package_info: PackageInfo) -> None:
        """Commit changes and push to remote repository.

        Args:
            package_info: Package information for commit message

        Raises:
            SystemExit: If git operations fail
        """
        if self.dry_run:
            self.log("Would commit and push changes:", "info")
            self.log(
                f"  Message: chore: update formula to {package_info.version}", "debug"
            )
            self.log(f"  File: {self.formula_name}", "debug")
            return

        # Check authentication
        if not self.github_token:
            self.log("HOMEBREW_TAP_TOKEN not set - push may fail", "warning")
            self.log(
                "Set HOMEBREW_TAP_TOKEN environment variable for authentication", "info"
            )

        try:
            # Configure git to use token
            if self.github_token:
                # Extract repo path from URL
                repo_match = re.search(
                    r"github\.com[:/](.+?)(?:\.git)?$", self.tap_repo_url
                )
                if repo_match:
                    repo_path = repo_match.group(1)
                    authenticated_url = (
                        f"https://{self.github_token}@github.com/{repo_path}.git"
                    )

                    subprocess.run(  # nosec B607
                        [
                            "git",
                            "-C",
                            str(self.tap_repo_path),
                            "remote",
                            "set-url",
                            "origin",
                            authenticated_url,
                        ],
                        capture_output=True,
                        check=True,
                    )
                    self.log("Configured git authentication", "debug")

            # Stage changes
            subprocess.run(  # nosec B607
                [
                    "git",
                    "-C",
                    str(self.tap_repo_path),
                    "add",
                    f"Formula/{self.formula_name}",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            # Try old location too if it exists
            old_formula = self.tap_repo_path / self.formula_name
            if old_formula.exists():
                subprocess.run(  # nosec B607
                    ["git", "-C", str(self.tap_repo_path), "add", self.formula_name],
                    capture_output=True,
                    text=True,
                    check=False,  # Don't fail if file not tracked
                )

            # Commit
            commit_message = f"chore: update formula to {package_info.version}\n\nUpdated mcp-vector-search to version {package_info.version}\n- Version: {package_info.version}\n- SHA256: {package_info.sha256}"

            subprocess.run(  # nosec B607
                ["git", "-C", str(self.tap_repo_path), "commit", "-m", commit_message],
                capture_output=True,
                text=True,
                check=True,
            )
            self.created_commit = True
            self.log("Changes committed", "success")

            # Push
            self.log("Pushing to remote repository...", "info")
            subprocess.run(  # nosec B607
                ["git", "-C", str(self.tap_repo_path), "push"],
                capture_output=True,
                text=True,
                check=True,
            )
            self.log("Changes pushed successfully", "success")

        except subprocess.CalledProcessError as e:
            self.log(f"Git operation failed: {e.stderr}", "error")

            if "authentication" in e.stderr.lower() or "permission" in e.stderr.lower():
                self.log("Authentication failed - check HOMEBREW_TAP_TOKEN", "error")
                sys.exit(5)
            else:
                self.rollback()
                sys.exit(2)

    def rollback(self) -> None:
        """Rollback changes if something went wrong."""
        if self.dry_run:
            return

        self.log("Rolling back changes...", "warning")

        # Restore backup
        if self.backup_path and self.backup_path.exists():
            formula_path = self.tap_repo_path / "Formula" / self.formula_name
            if not formula_path.exists():
                formula_path = self.tap_repo_path / self.formula_name

            shutil.copy2(self.backup_path, formula_path)
            self.log("Restored formula from backup", "success")

        # Reset git if commit was created
        if self.created_commit:
            try:
                subprocess.run(  # nosec B607
                    ["git", "-C", str(self.tap_repo_path), "reset", "--hard", "HEAD~1"],
                    capture_output=True,
                    check=True,
                )
                self.log("Reset git commit", "success")
            except subprocess.CalledProcessError:
                self.log("Failed to reset git commit", "warning")

    def cleanup(self) -> None:
        """Cleanup backup files."""
        if self.backup_path and self.backup_path.exists():
            self.backup_path.unlink()
            self.log("Removed backup file", "debug")

    def run(self, version: str | None = None) -> None:
        """Run the complete update process.

        Args:
            version: Specific version to update to, or None for latest
        """
        try:
            # Header
            print(f"\n{BOLD}{BLUE}{'=' * 60}{RESET}")
            print(f"{BOLD}{BLUE}Homebrew Formula Updater for mcp-vector-search{RESET}")
            print(f"{BOLD}{BLUE}{'=' * 60}{RESET}\n")

            # Fetch PyPI info
            package_info = self.fetch_pypi_info(version)

            # Verify hash
            if not self.verify_sha256(package_info):
                self.log("Hash verification failed", "error")
                sys.exit(4)

            # Setup repository
            self.setup_tap_repository()

            # Update formula
            formula_path = self.update_formula(package_info)

            # Validate
            if not self.validate_formula(formula_path):
                self.log("Formula validation failed", "error")
                self.rollback()
                sys.exit(4)

            # Commit and push
            self.commit_and_push(package_info)

            # Cleanup
            self.cleanup()

            # Success message
            print(f"\n{BOLD}{GREEN}{'=' * 60}{RESET}")
            print(f"{BOLD}{GREEN}✓ Formula updated successfully!{RESET}")
            print(f"{BOLD}{GREEN}{'=' * 60}{RESET}\n")

            if not self.dry_run:
                self.log(
                    "Users can now install with: brew install bobmatnyc/mcp-vector-search/mcp-vector-search",
                    "info",
                )

        except KeyboardInterrupt:
            print(f"\n{YELLOW}Operation cancelled by user{RESET}")
            self.rollback()
            sys.exit(130)
        except Exception as e:
            self.log(f"Unexpected error: {e}", "error")
            self.rollback()
            sys.exit(1)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Update Homebrew formula for mcp-vector-search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dry-run                    # Test update to latest version
  %(prog)s                              # Update to latest version
  %(prog)s --version 0.12.8             # Update to specific version
  %(prog)s --tap-repo-path /custom/path # Use custom tap repo path
  %(prog)s --verbose                    # Show detailed output

Environment Variables:
  HOMEBREW_TAP_TOKEN      GitHub personal access token for authentication
  HOMEBREW_TAP_REPO       Custom tap repository URL

Exit Codes:
  0: Success
  1: PyPI API error
  2: Git operation error
  3: Formula update error
  4: Validation error
  5: Authentication error
        """,
    )

    parser.add_argument(
        "--version",
        metavar="VERSION",
        help="Specific version to update to (default: latest from PyPI)",
    )

    parser.add_argument(
        "--tap-repo-path",
        type=Path,
        metavar="PATH",
        help="Local path to tap repository (default: ~/.homebrew_tap_update/homebrew-mcp-vector-search)",
    )

    parser.add_argument(
        "--tap-repo-url",
        metavar="URL",
        help="URL of tap repository (default: from HOMEBREW_TAP_REPO or bobmatnyc/homebrew-mcp-vector-search)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Create updater
    updater = HomebrewFormulaUpdater(
        tap_repo_path=args.tap_repo_path,
        tap_repo_url=args.tap_repo_url,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    # Run update
    updater.run(version=args.version)


if __name__ == "__main__":
    main()
