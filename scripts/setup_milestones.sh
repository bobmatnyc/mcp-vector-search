#!/bin/bash
# Script to create GitHub milestones and assign issues for mcp-vector-search
# Usage: ./scripts/setup_milestones.sh

set -e

REPO="bobmatnyc/mcp-vector-search"

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Creating GitHub milestones for ${REPO}${NC}\n"

# Calculate due dates
DUE_DATE_1=$(date -v+2w +%Y-%m-%dT23:59:59Z)
DUE_DATE_2=$(date -v+3w +%Y-%m-%dT23:59:59Z)
DUE_DATE_3=$(date -v+4w +%Y-%m-%dT23:59:59Z)
DUE_DATE_4=$(date -v+5w +%Y-%m-%dT23:59:59Z)
DUE_DATE_5=$(date -v+8w +%Y-%m-%dT23:59:59Z)

# Create Milestone 1: v0.17.0 - Core Metrics
echo -e "${GREEN}Creating Milestone 1: v0.17.0 - Core Metrics${NC}"
MILESTONE_1=$(gh api repos/${REPO}/milestones -X POST \
  -f title="v0.17.0 - Core Metrics" \
  -f description="Tier 1 collectors integrated into indexer, extended chunk metadata in ChromaDB, analyze --quick command, basic console reporter" \
  -f due_on="${DUE_DATE_1}" \
  -f state="open" | jq -r '.number')
echo "Created milestone ${MILESTONE_1}"

# Create Milestone 2: v0.18.0 - Quality Gates
echo -e "${GREEN}Creating Milestone 2: v0.18.0 - Quality Gates${NC}"
MILESTONE_2=$(gh api repos/${REPO}/milestones -X POST \
  -f title="v0.18.0 - Quality Gates" \
  -f description="Threshold configuration system, SARIF output for CI integration, --fail-on-smell exit codes, diff-aware analysis" \
  -f due_on="${DUE_DATE_2}" \
  -f state="open" | jq -r '.number')
echo "Created milestone ${MILESTONE_2}"

# Create Milestone 3: v0.19.0 - Cross-File Analysis
echo -e "${GREEN}Creating Milestone 3: v0.19.0 - Cross-File Analysis${NC}"
MILESTONE_3=$(gh api repos/${REPO}/milestones -X POST \
  -f title="v0.19.0 - Cross-File Analysis" \
  -f description="Tier 4 collectors (afferent coupling, circular deps), dependency graph construction, SQLite metrics store, trend tracking" \
  -f due_on="${DUE_DATE_3}" \
  -f state="open" | jq -r '.number')
echo "Created milestone ${MILESTONE_3}"

# Create Milestone 4: v0.20.0 - Visualization Export
echo -e "${GREEN}Creating Milestone 4: v0.20.0 - Visualization Export${NC}"
MILESTONE_4=$(gh api repos/${REPO}/milestones -X POST \
  -f title="v0.20.0 - Visualization Export" \
  -f description="JSON export for visualizer, all chart data schemas finalized, HTML standalone report, documentation" \
  -f due_on="${DUE_DATE_4}" \
  -f state="open" | jq -r '.number')
echo "Created milestone ${MILESTONE_4}"

# Create Milestone 5: v0.21.0 - Search Integration
echo -e "${GREEN}Creating Milestone 5: v0.21.0 - Search Integration${NC}"
MILESTONE_5=$(gh api repos/${REPO}/milestones -X POST \
  -f title="v0.21.0 - Search Integration" \
  -f description="Quality-aware search ranking and filtering, MCP tool exposure" \
  -f due_on="${DUE_DATE_5}" \
  -f state="open" | jq -r '.number')
echo "Created milestone ${MILESTONE_5}"

echo -e "\n${BLUE}Assigning issues to milestones${NC}\n"

# Milestone 1 issues (#1-11)
echo -e "${GREEN}Assigning issues to Milestone 1 (Core Metrics)${NC}"
for issue in 1 2 3 4 5 6 7 8 9 10 11; do
  gh api repos/${REPO}/issues/${issue} -X PATCH -f milestone=${MILESTONE_1}
  echo "  Assigned #${issue} to milestone ${MILESTONE_1}"
done

# Milestone 2 issues (#12-18)
echo -e "${GREEN}Assigning issues to Milestone 2 (Quality Gates)${NC}"
for issue in 12 13 14 15 16 17 18; do
  gh api repos/${REPO}/issues/${issue} -X PATCH -f milestone=${MILESTONE_2}
  echo "  Assigned #${issue} to milestone ${MILESTONE_2}"
done

# Milestone 3 issues (#19-26)
echo -e "${GREEN}Assigning issues to Milestone 3 (Cross-File Analysis)${NC}"
for issue in 19 20 21 22 23 24 25 26; do
  gh api repos/${REPO}/issues/${issue} -X PATCH -f milestone=${MILESTONE_3}
  echo "  Assigned #${issue} to milestone ${MILESTONE_3}"
done

# Milestone 4 issues (#27-33)
echo -e "${GREEN}Assigning issues to Milestone 4 (Visualization Export)${NC}"
for issue in 27 28 29 30 31 32 33; do
  gh api repos/${REPO}/issues/${issue} -X PATCH -f milestone=${MILESTONE_4}
  echo "  Assigned #${issue} to milestone ${MILESTONE_4}"
done

# Milestone 5 issues (#34-37)
echo -e "${GREEN}Assigning issues to Milestone 5 (Search Integration)${NC}"
for issue in 34 35 36 37; do
  gh api repos/${REPO}/issues/${issue} -X PATCH -f milestone=${MILESTONE_5}
  echo "  Assigned #${issue} to milestone ${MILESTONE_5}"
done

echo -e "\n${BLUE}Milestones created and issues assigned successfully!${NC}"
echo -e "View at: https://github.com/${REPO}/milestones"
