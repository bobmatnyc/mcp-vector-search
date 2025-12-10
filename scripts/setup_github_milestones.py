#!/usr/bin/env python3
"""
Setup GitHub milestones and issue dependencies for mcp-vector-search project.

This script uses the PyGithub library to interact with the GitHub API.
Install with: pip install PyGithub

Usage:
    python scripts/setup_github_milestones.py --token YOUR_GITHUB_TOKEN
    OR set GITHUB_TOKEN environment variable
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

try:
    from github import Github
    from github.GithubException import GithubException
except ImportError:
    print("Error: PyGithub not installed. Install with: pip install PyGithub")
    sys.exit(1)

# Repository details
REPO_OWNER = "bobmatnyc"
REPO_NAME = "mcp-vector-search"

# Milestone definitions
MILESTONES = [
    {
        "title": "v0.17.0 - Core Metrics",
        "description": "Tier 1 collectors integrated into indexer, extended chunk metadata in ChromaDB, analyze --quick command, basic console reporter",
        "due_weeks": 2,
        "issues": list(range(1, 12)),  # Issues #1-11
    },
    {
        "title": "v0.18.0 - Quality Gates",
        "description": "Threshold configuration system, SARIF output for CI integration, --fail-on-smell exit codes, diff-aware analysis",
        "due_weeks": 3,
        "issues": list(range(12, 19)),  # Issues #12-18
    },
    {
        "title": "v0.19.0 - Cross-File Analysis",
        "description": "Tier 4 collectors (afferent coupling, circular deps), dependency graph construction, SQLite metrics store, trend tracking",
        "due_weeks": 4,
        "issues": list(range(19, 27)),  # Issues #19-26
    },
    {
        "title": "v0.20.0 - Visualization Export",
        "description": "JSON export for visualizer, all chart data schemas finalized, HTML standalone report, documentation",
        "due_weeks": 5,
        "issues": list(range(27, 34)),  # Issues #27-33
    },
    {
        "title": "v0.21.0 - Search Integration",
        "description": "Quality-aware search ranking and filtering, MCP tool exposure",
        "due_weeks": 8,
        "issues": list(range(34, 38)),  # Issues #34-37
    },
]

# Issue dependencies
DEPENDENCIES = {
    2: {"blocked_by": [], "blocks": [3, 4, 5, 6, 7, 8, 9]},
    3: {"blocked_by": [2], "blocks": [8]},
    4: {"blocked_by": [2], "blocks": [8]},
    5: {"blocked_by": [2], "blocks": [8]},
    6: {"blocked_by": [2], "blocks": [8]},
    7: {"blocked_by": [2], "blocks": [8]},
    8: {"blocked_by": [2, 3, 4, 5, 6, 7], "blocks": [10, 14, 20, 26, 31]},
    9: {"blocked_by": [2], "blocks": [10]},
    10: {"blocked_by": [8, 9], "blocks": [11, 14, 15, 17, 29, 33, 35]},
    11: {"blocked_by": [10], "blocks": []},
    13: {"blocked_by": [2], "blocks": [14]},
    14: {"blocked_by": [8, 13], "blocks": [15, 16, 32, 35]},
    15: {"blocked_by": [10, 14], "blocks": [16]},
    16: {"blocked_by": [14, 15], "blocks": []},
    17: {"blocked_by": [10], "blocks": [18]},
    18: {"blocked_by": [17], "blocks": []},
    20: {"blocked_by": [2, 8], "blocks": [21, 22, 23]},
    21: {"blocked_by": [20], "blocks": [22]},
    22: {"blocked_by": [20, 21], "blocks": []},
    23: {"blocked_by": [20], "blocks": []},
    24: {"blocked_by": [2], "blocks": [25, 32, 33]},
    25: {"blocked_by": [24], "blocks": []},
    26: {"blocked_by": [2, 8], "blocks": []},
    28: {"blocked_by": [2], "blocks": [29]},
    29: {"blocked_by": [28, 10], "blocks": [30]},
    30: {"blocked_by": [29], "blocks": []},
    31: {"blocked_by": [2, 8], "blocks": []},
    32: {"blocked_by": [14, 24], "blocks": []},
    33: {"blocked_by": [10, 24], "blocks": []},
    35: {"blocked_by": [10, 14], "blocks": [36, 37]},
    36: {"blocked_by": [35], "blocks": []},
    37: {"blocked_by": [10, 35], "blocks": []},
}


def create_milestones(repo, dry_run=False):
    """Create milestones in the repository."""
    print("\n=== Creating Milestones ===\n")
    created_milestones = {}

    for milestone_def in MILESTONES:
        title = milestone_def["title"]
        description = milestone_def["description"]
        due_date = datetime.now() + timedelta(weeks=milestone_def["due_weeks"])

        print(f"Creating milestone: {title}")
        print(f"  Description: {description}")
        print(f"  Due date: {due_date.strftime('%Y-%m-%d')}")

        if dry_run:
            print("  [DRY RUN] Would create milestone")
            created_milestones[title] = None
        else:
            try:
                milestone = repo.create_milestone(
                    title=title, description=description, due_on=due_date, state="open"
                )
                created_milestones[title] = milestone
                print(f"  ✓ Created milestone #{milestone.number}")
            except GithubException as e:
                print(f"  ✗ Error creating milestone: {e}")

        print()

    return created_milestones


def assign_issues_to_milestones(repo, milestones, dry_run=False):
    """Assign issues to their respective milestones."""
    print("\n=== Assigning Issues to Milestones ===\n")

    for milestone_def in MILESTONES:
        title = milestone_def["title"]
        milestone = milestones.get(title)

        if milestone is None and not dry_run:
            print(f"Skipping {title} (not created)")
            continue

        print(f"Assigning issues to {title}:")

        for issue_num in milestone_def["issues"]:
            if dry_run:
                print(f"  [DRY RUN] Would assign issue #{issue_num}")
            else:
                try:
                    issue = repo.get_issue(issue_num)
                    issue.edit(milestone=milestone)
                    print(f"  ✓ Assigned issue #{issue_num}: {issue.title}")
                except GithubException as e:
                    print(f"  ✗ Error assigning issue #{issue_num}: {e}")

        print()


def add_dependencies_to_issues(repo, dry_run=False):
    """Add dependency information to issue descriptions."""
    print("\n=== Adding Dependencies to Issues ===\n")

    for issue_num, deps in DEPENDENCIES.items():
        blocked_by = deps["blocked_by"]
        blocks = deps["blocks"]

        print(f"Updating issue #{issue_num}:")

        if dry_run:
            print("  [DRY RUN] Would add dependencies:")
            print(f"    Blocked by: {blocked_by or 'None'}")
            print(f"    Blocks: {blocks or 'None'}")
        else:
            try:
                issue = repo.get_issue(issue_num)
                current_body = issue.body or ""

                # Check if dependencies section already exists
                if "## Dependencies" in current_body:
                    print("  ⊘ Skipped (already has dependencies)")
                    continue

                # Build dependency section
                dep_section = "\n\n## Dependencies\n\n"

                if blocked_by:
                    blocked_by_str = ", ".join([f"#{i}" for i in blocked_by])
                    dep_section += f"**Blocked by:** {blocked_by_str}\n"
                else:
                    dep_section += "**Blocked by:** None (can start immediately)\n"

                if blocks:
                    blocks_str = ", ".join([f"#{i}" for i in blocks])
                    dep_section += f"**Blocks:** {blocks_str}\n"

                # Update issue
                new_body = current_body + dep_section
                issue.edit(body=new_body)
                print(f"  ✓ Added dependencies to issue #{issue_num}")

            except GithubException as e:
                print(f"  ✗ Error updating issue #{issue_num}: {e}")

        print()


def main():
    parser = argparse.ArgumentParser(
        description="Setup GitHub milestones and issue dependencies"
    )
    parser.add_argument(
        "--token",
        help="GitHub personal access token (or set GITHUB_TOKEN env var)",
        default=os.getenv("GITHUB_TOKEN"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--skip-milestones",
        action="store_true",
        help="Skip milestone creation (only add dependencies)",
    )
    parser.add_argument(
        "--skip-dependencies",
        action="store_true",
        help="Skip dependency updates (only create milestones)",
    )

    args = parser.parse_args()

    if not args.token:
        print("Error: GitHub token required. Use --token or set GITHUB_TOKEN env var")
        sys.exit(1)

    print(f"Connecting to GitHub repository: {REPO_OWNER}/{REPO_NAME}")

    try:
        g = Github(args.token)
        repo = g.get_repo(f"{REPO_OWNER}/{REPO_NAME}")
        print(f"✓ Connected to repository: {repo.full_name}")
    except GithubException as e:
        print(f"✗ Error connecting to repository: {e}")
        sys.exit(1)

    if args.dry_run:
        print("\n[DRY RUN MODE] No changes will be made\n")

    # Create milestones
    milestones = {}
    if not args.skip_milestones:
        milestones = create_milestones(repo, dry_run=args.dry_run)
        if not args.dry_run:
            assign_issues_to_milestones(repo, milestones, dry_run=args.dry_run)

    # Add dependencies
    if not args.skip_dependencies:
        add_dependencies_to_issues(repo, dry_run=args.dry_run)

    print("\n=== Summary ===")
    print(f"Milestones: {len(MILESTONES)} defined")
    print(f"Issues: {sum(len(m['issues']) for m in MILESTONES)} to be assigned")
    print(f"Dependencies: {len(DEPENDENCIES)} issues with dependencies")
    print(f"\nView milestones: https://github.com/{REPO_OWNER}/{REPO_NAME}/milestones")
    print(f"View issues: https://github.com/{REPO_OWNER}/{REPO_NAME}/issues")


if __name__ == "__main__":
    main()
