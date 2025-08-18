#!/bin/bash
# ============================================================================
# DEPRECATION NOTICE: This script is deprecated as of v4.0.3
# Please use the Makefile instead:
#   make build-package    # Build distribution packages
#   make version-patch    # Bump patch version
#   make release-patch    # Full release workflow
# 
# This script will be removed in v5.0.0
# ============================================================================

"""
Quick development build script with automatic version increment.

Usage:
  ./scripts/build.sh           # Increment patch version and rebuild
  ./scripts/build.sh minor     # Increment minor version and rebuild  
  ./scripts/build.sh major     # Increment major version and rebuild
  ./scripts/build.sh --no-inc  # Just rebuild without version increment
"""

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Parse arguments
INCREMENT_TYPE="patch"
NO_INCREMENT=false

case "${1:-}" in
    "major"|"minor"|"patch")
        INCREMENT_TYPE="$1"
        ;;
    "--no-inc"|"--no-increment")
        NO_INCREMENT=true
        ;;
    "--help"|"-h")
        echo "Usage: $0 [major|minor|patch|--no-inc]"
        echo "  major     - Increment major version (x.0.0)"
        echo "  minor     - Increment minor version (0.x.0)" 
        echo "  patch     - Increment patch version (0.0.x) [default]"
        echo "  --no-inc  - Skip version increment, just rebuild"
        exit 0
        ;;
    "")
        # Default to patch
        ;;
    *)
        echo "Error: Unknown argument '$1'"
        echo "Use --help for usage information"
        exit 1
        ;;
esac

# Run the Python script
if [ "$NO_INCREMENT" = true ]; then
    python3 scripts/dev-build.py --no-increment
else
    python3 scripts/dev-build.py --increment "$INCREMENT_TYPE"
fi
