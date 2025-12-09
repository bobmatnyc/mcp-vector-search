#!/bin/bash
# Full Release Script for mcp-vector-search
# Handles: Version bump, PyPI publish, pipx upgrade, Homebrew update
#
# Usage:
#   ./scripts/full_release.sh [patch|minor|major] [--dry-run]
#
# Examples:
#   ./scripts/full_release.sh patch           # Bump patch version and publish
#   ./scripts/full_release.sh minor --dry-run # Preview minor release
#   ./scripts/full_release.sh                 # Just publish current version (no bump)

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Parse arguments
BUMP_TYPE=""
DRY_RUN=false
SKIP_HOMEBREW=false
SKIP_PIPX=false

for arg in "$@"; do
    case $arg in
        patch|minor|major)
            BUMP_TYPE="$arg"
            ;;
        --dry-run)
            DRY_RUN=true
            ;;
        --skip-homebrew)
            SKIP_HOMEBREW=true
            ;;
        --skip-pipx)
            SKIP_PIPX=true
            ;;
        --help|-h)
            echo "Usage: $0 [patch|minor|major] [--dry-run] [--skip-homebrew] [--skip-pipx]"
            echo ""
            echo "Arguments:"
            echo "  patch|minor|major  Version bump type (optional, skips bump if not provided)"
            echo "  --dry-run          Preview changes without making them"
            echo "  --skip-homebrew    Skip Homebrew formula update"
            echo "  --skip-pipx        Skip pipx upgrade verification"
            exit 0
            ;;
    esac
done

# Get current version
get_version() {
    python3 -c "
import sys
sys.path.insert(0, '$PROJECT_DIR/src')
from mcp_vector_search import __version__
print(__version__)
"
}

# Get build number
get_build() {
    python3 -c "
import sys
sys.path.insert(0, '$PROJECT_DIR/src')
from mcp_vector_search import __build__
print(__build__)
"
}

echo ""
echo -e "${BOLD}${BLUE}============================================${NC}"
echo -e "${BOLD}${BLUE}  mcp-vector-search Full Release Script     ${NC}"
echo -e "${BOLD}${BLUE}============================================${NC}"
echo ""

if $DRY_RUN; then
    echo -e "${YELLOW}[DRY RUN MODE] No changes will be made${NC}"
    echo ""
fi

# Change to project directory
cd "$PROJECT_DIR"

# Current version
CURRENT_VERSION=$(get_version)
CURRENT_BUILD=$(get_build)
echo -e "${CYAN}Current Version:${NC} $CURRENT_VERSION (build $CURRENT_BUILD)"

# ============================================================================
# STEP 1: Version Bump (if requested)
# ============================================================================
if [ -n "$BUMP_TYPE" ]; then
    echo ""
    echo -e "${BOLD}${BLUE}Step 1: Version Bump ($BUMP_TYPE)${NC}"
    echo "-------------------------------------------"

    if $DRY_RUN; then
        echo -e "${YELLOW}Would bump version from $CURRENT_VERSION${NC}"
    else
        python3 "$SCRIPT_DIR/version_manager.py" --bump "$BUMP_TYPE"
        NEW_VERSION=$(get_version)
        NEW_BUILD=$(get_build)
        echo -e "${GREEN}✓ Version bumped: $CURRENT_VERSION → $NEW_VERSION (build $NEW_BUILD)${NC}"
        CURRENT_VERSION=$NEW_VERSION
        CURRENT_BUILD=$NEW_BUILD
    fi
else
    echo ""
    echo -e "${CYAN}Step 1: Skipping version bump (using current: $CURRENT_VERSION)${NC}"
fi

# ============================================================================
# STEP 2: Quality Checks
# ============================================================================
echo ""
echo -e "${BOLD}${BLUE}Step 2: Quality Checks${NC}"
echo "-------------------------------------------"

if $DRY_RUN; then
    echo -e "${YELLOW}Would run: make pre-publish${NC}"
else
    echo "Running quality checks..."
    make pre-publish || {
        echo -e "${RED}✗ Quality checks failed${NC}"
        exit 1
    }
    echo -e "${GREEN}✓ All quality checks passed${NC}"
fi

# ============================================================================
# STEP 3: Build Package
# ============================================================================
echo ""
echo -e "${BOLD}${BLUE}Step 3: Build Package${NC}"
echo "-------------------------------------------"

if $DRY_RUN; then
    echo -e "${YELLOW}Would run: python -m build${NC}"
else
    echo "Building package..."
    rm -rf dist/ build/ *.egg-info
    python -m build
    echo -e "${GREEN}✓ Package built${NC}"
    ls -la dist/
fi

# ============================================================================
# STEP 4: Publish to PyPI
# ============================================================================
echo ""
echo -e "${BOLD}${BLUE}Step 4: Publish to PyPI${NC}"
echo "-------------------------------------------"

if $DRY_RUN; then
    echo -e "${YELLOW}Would run: twine upload dist/*${NC}"
else
    echo "Publishing to PyPI..."
    python -m twine upload dist/* || {
        echo -e "${RED}✗ PyPI publish failed${NC}"
        exit 1
    }
    echo -e "${GREEN}✓ Published to PyPI: https://pypi.org/project/mcp-vector-search/$CURRENT_VERSION/${NC}"
fi

# ============================================================================
# STEP 5: Wait for PyPI availability
# ============================================================================
echo ""
echo -e "${BOLD}${BLUE}Step 5: Verify PyPI Availability${NC}"
echo "-------------------------------------------"

if $DRY_RUN; then
    echo -e "${YELLOW}Would wait for PyPI to process the package${NC}"
else
    echo "Waiting for PyPI to process the package..."
    MAX_WAIT=120
    WAIT_INTERVAL=5
    ELAPSED=0

    while [ $ELAPSED -lt $MAX_WAIT ]; do
        if curl -s -f -o /dev/null "https://pypi.org/pypi/mcp-vector-search/$CURRENT_VERSION/json"; then
            echo -e "${GREEN}✓ Version $CURRENT_VERSION is available on PyPI${NC}"
            break
        fi
        echo "  Waiting... ($ELAPSED/$MAX_WAIT seconds)"
        sleep $WAIT_INTERVAL
        ELAPSED=$((ELAPSED + WAIT_INTERVAL))
    done

    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo -e "${YELLOW}⚠ PyPI availability check timed out, continuing anyway...${NC}"
    fi
fi

# ============================================================================
# STEP 6: Upgrade via pipx
# ============================================================================
if ! $SKIP_PIPX; then
    echo ""
    echo -e "${BOLD}${BLUE}Step 6: Upgrade via pipx${NC}"
    echo "-------------------------------------------"

    if $DRY_RUN; then
        echo -e "${YELLOW}Would run: pipx upgrade mcp-vector-search --force${NC}"
    else
        echo "Upgrading local pipx installation..."
        pipx upgrade mcp-vector-search --force || {
            echo -e "${YELLOW}⚠ pipx upgrade failed, trying fresh install...${NC}"
            pipx install mcp-vector-search --force || {
                echo -e "${YELLOW}⚠ pipx install also failed, continuing...${NC}"
            }
        }

        # Verify installation
        INSTALLED_VERSION=$(mcp-vector-search --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || echo "unknown")
        if [ "$INSTALLED_VERSION" = "$CURRENT_VERSION" ]; then
            echo -e "${GREEN}✓ pipx installation verified: $INSTALLED_VERSION${NC}"
        else
            echo -e "${YELLOW}⚠ Version mismatch: expected $CURRENT_VERSION, got $INSTALLED_VERSION${NC}"
        fi
    fi
else
    echo ""
    echo -e "${CYAN}Step 6: Skipping pipx upgrade (--skip-pipx)${NC}"
fi

# ============================================================================
# STEP 7: Update Homebrew Formula
# ============================================================================
if ! $SKIP_HOMEBREW; then
    echo ""
    echo -e "${BOLD}${BLUE}Step 7: Update Homebrew Formula${NC}"
    echo "-------------------------------------------"

    if [ -z "$HOMEBREW_TAP_TOKEN" ]; then
        echo -e "${YELLOW}⚠ HOMEBREW_TAP_TOKEN not set, skipping Homebrew update${NC}"
        echo "  To enable, export HOMEBREW_TAP_TOKEN=<your-github-token>"
    else
        if $DRY_RUN; then
            echo -e "${YELLOW}Would run: python3 scripts/update_homebrew_formula.py --version $CURRENT_VERSION${NC}"
        else
            echo "Updating Homebrew formula..."
            python3 "$SCRIPT_DIR/update_homebrew_formula.py" --version "$CURRENT_VERSION" --verbose || {
                echo -e "${YELLOW}⚠ Homebrew formula update failed${NC}"
                echo "  Run manually: python3 scripts/update_homebrew_formula.py --version $CURRENT_VERSION --verbose"
            }
        fi
    fi
else
    echo ""
    echo -e "${CYAN}Step 7: Skipping Homebrew update (--skip-homebrew)${NC}"
fi

# ============================================================================
# STEP 8: Create Git Tag & GitHub Release
# ============================================================================
echo ""
echo -e "${BOLD}${BLUE}Step 8: Git Tag & GitHub Release${NC}"
echo "-------------------------------------------"

if $DRY_RUN; then
    echo -e "${YELLOW}Would create git tag: v$CURRENT_VERSION${NC}"
    echo -e "${YELLOW}Would create GitHub release${NC}"
else
    # Check if tag exists
    if git tag -l "v$CURRENT_VERSION" | grep -q "v$CURRENT_VERSION"; then
        echo -e "${CYAN}Tag v$CURRENT_VERSION already exists${NC}"
    else
        echo "Creating git tag..."
        git tag -a "v$CURRENT_VERSION" -m "Release v$CURRENT_VERSION"
        git push origin "v$CURRENT_VERSION"
        echo -e "${GREEN}✓ Git tag created: v$CURRENT_VERSION${NC}"
    fi

    # Create GitHub release
    if command -v gh &> /dev/null; then
        echo "Creating GitHub release..."
        gh release create "v$CURRENT_VERSION" \
            --title "v$CURRENT_VERSION" \
            --generate-notes \
            dist/* 2>/dev/null || {
            echo -e "${YELLOW}⚠ GitHub release may already exist or creation failed${NC}"
        }
        echo -e "${GREEN}✓ GitHub release created (or already exists)${NC}"
    else
        echo -e "${YELLOW}⚠ GitHub CLI not found, skipping GitHub release${NC}"
    fi
fi

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo -e "${BOLD}${GREEN}============================================${NC}"
echo -e "${BOLD}${GREEN}  Release Complete: v$CURRENT_VERSION       ${NC}"
echo -e "${BOLD}${GREEN}============================================${NC}"
echo ""
echo -e "${CYAN}Links:${NC}"
echo "  PyPI:    https://pypi.org/project/mcp-vector-search/$CURRENT_VERSION/"
echo "  GitHub:  https://github.com/bobmatnyc/mcp-vector-search/releases/tag/v$CURRENT_VERSION"
if [ -n "$HOMEBREW_TAP_TOKEN" ]; then
    echo "  Homebrew: brew install bobmatnyc/mcp-vector-search/mcp-vector-search"
fi
echo ""
echo -e "${CYAN}Install/Upgrade Commands:${NC}"
echo "  pip:     pip install mcp-vector-search==$CURRENT_VERSION"
echo "  pipx:    pipx upgrade mcp-vector-search --force"
echo "  brew:    brew upgrade mcp-vector-search"
echo ""

if $DRY_RUN; then
    echo -e "${YELLOW}[DRY RUN] No changes were made${NC}"
fi
