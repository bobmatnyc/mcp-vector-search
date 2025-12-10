#!/bin/bash
# Script to add dependency information to GitHub issues
# Usage: ./scripts/add_issue_dependencies.sh

set -e

REPO="bobmatnyc/mcp-vector-search"

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}Adding dependency information to issues${NC}\n"

# Function to add dependencies to an issue
add_dependencies() {
  local issue_number=$1
  local blocked_by=$2
  local blocks=$3

  echo -e "${GREEN}Updating issue #${issue_number}${NC}"

  # Get current issue body
  current_body=$(gh api repos/${REPO}/issues/${issue_number} -q '.body')

  # Build dependency section
  dep_section="\n\n## Dependencies\n"

  if [ -n "${blocked_by}" ]; then
    dep_section="${dep_section}\n**Blocked by:** ${blocked_by}"
  else
    dep_section="${dep_section}\n**Blocked by:** None (can start immediately)"
  fi

  if [ -n "${blocks}" ]; then
    dep_section="${dep_section}\n**Blocks:** ${blocks}"
  fi

  # Append dependencies if not already present
  if [[ $current_body != *"## Dependencies"* ]]; then
    new_body="${current_body}${dep_section}"
    gh api repos/${REPO}/issues/${issue_number} -X PATCH -f body="${new_body}"
    echo "  Added dependencies: blocked_by=[${blocked_by:-none}], blocks=[${blocks:-none}]"
  else
    echo -e "  ${YELLOW}Skipped (already has dependencies section)${NC}"
  fi
}

# Phase 1 Dependencies (Milestone 1)
echo -e "\n${BLUE}Phase 1: Core Metrics${NC}"
add_dependencies 2 "" "#3, #4, #5, #6, #7, #8, #9"
add_dependencies 3 "#2" "#8"
add_dependencies 4 "#2" "#8"
add_dependencies 5 "#2" "#8"
add_dependencies 6 "#2" "#8"
add_dependencies 7 "#2" "#8"
add_dependencies 8 "#2, #3, #4, #5, #6, #7" "#10, #14, #20, #26, #31"
add_dependencies 9 "#2" "#10"
add_dependencies 10 "#8, #9" "#11, #14, #15, #17, #29, #33, #35"
add_dependencies 11 "#10" ""

# Phase 2 Dependencies (Milestone 2)
echo -e "\n${BLUE}Phase 2: Quality Gates${NC}"
add_dependencies 13 "#2" "#14"
add_dependencies 14 "#8, #13" "#15, #16, #32, #35"
add_dependencies 15 "#10, #14" "#16"
add_dependencies 16 "#14, #15" ""
add_dependencies 17 "#10" "#18"
add_dependencies 18 "#17" ""

# Phase 3 Dependencies (Milestone 3)
echo -e "\n${BLUE}Phase 3: Cross-File Analysis${NC}"
add_dependencies 20 "#2, #8" "#21, #22, #23"
add_dependencies 21 "#20" "#22"
add_dependencies 22 "#20, #21" ""
add_dependencies 23 "#20" ""
add_dependencies 24 "#2" "#25, #32, #33"
add_dependencies 25 "#24" ""
add_dependencies 26 "#2, #8" ""

# Phase 4 Dependencies (Milestone 4)
echo -e "\n${BLUE}Phase 4: Visualization Export${NC}"
add_dependencies 28 "#2" "#29"
add_dependencies 29 "#28, #10" "#30"
add_dependencies 30 "#29" ""
add_dependencies 31 "#2, #8" ""
add_dependencies 32 "#14, #24" ""
add_dependencies 33 "#10, #24" ""

# Phase 5 Dependencies (Milestone 5)
echo -e "\n${BLUE}Phase 5: Search Integration${NC}"
add_dependencies 35 "#10, #14" "#36, #37"
add_dependencies 36 "#35" ""
add_dependencies 37 "#10, #35" ""

echo -e "\n${GREEN}Dependency information added successfully!${NC}"
echo -e "View issues at: https://github.com/${REPO}/issues"
