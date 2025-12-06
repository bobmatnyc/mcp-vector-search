#!/bin/bash
# Update git submodules to latest versions
#
# This script updates all git submodules to their latest commits from the remote repository.
# It's part of the MCP Vector Search build system.
#
# Usage:
#   ./scripts/update_submodules.sh
#
# Or via Make:
#   make submodule-update

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RESET='\033[0m'

echo -e "${BLUE}🔄 Updating git submodules...${RESET}"
echo ""

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo -e "${RED}✗ Error: Not in a git repository${RESET}"
    exit 1
fi

# Check if .gitmodules exists
if [ ! -f ".gitmodules" ]; then
    echo -e "${YELLOW}⚠️  No .gitmodules file found${RESET}"
    echo "   This project may not have any submodules configured."
    exit 0
fi

# Initialize submodules if not already done
echo -e "${BLUE}📦 Initializing submodules...${RESET}"
git submodule update --init --recursive

# Update to latest commits
echo -e "${BLUE}⬆️  Updating to latest remote versions...${RESET}"
git submodule update --remote

# Show submodule status
echo ""
echo -e "${BLUE}📊 Submodule Status:${RESET}"
git submodule status

# Check if submodules have changes
if git status --porcelain | grep -q "^ M"; then
    echo ""
    echo -e "${GREEN}✓ Submodules updated successfully${RESET}"
    echo ""
    echo -e "${YELLOW}📝 Changes detected in submodules:${RESET}"
    git status --short | grep "^ M"
    echo ""
    echo -e "${YELLOW}💡 To commit submodule updates:${RESET}"
    echo "   git add vendor/py-mcp-installer-service"
    echo "   git commit -m 'chore: update py-mcp-installer-service submodule'"
    echo ""
else
    echo ""
    echo -e "${GREEN}✓ Submodules are already up to date${RESET}"
fi

# Detailed status for each submodule
echo ""
echo -e "${BLUE}📋 Detailed Submodule Information:${RESET}"
echo ""

git submodule foreach --quiet '
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Submodule: $name"
    echo "Path: $sm_path"
    echo "URL: $(git config --get remote.origin.url)"
    echo "Branch: $(git rev-parse --abbrev-ref HEAD)"
    echo "Commit: $(git rev-parse --short HEAD)"
    echo "Message: $(git log -1 --pretty=format:"%s")"
    echo ""
'

echo -e "${GREEN}✓ Submodule update complete${RESET}"
