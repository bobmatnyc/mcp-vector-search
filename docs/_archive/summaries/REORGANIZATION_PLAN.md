# Documentation Reorganization Plan

## Executive Summary

This document outlines the complete reorganization of the mcp-vector-search documentation to create a clear, user-friendly structure that separates concerns and improves navigation.

**Date**: 2025-11-15
**Status**: In Progress
**Goal**: Create intuitive documentation hierarchy with clear separation between user and developer docs

---

## Current State Analysis

### Documentation Inventory (37 files)

#### Root Level (2 files)
- `README.md` - Main project overview
- `CLAUDE.md` - Project memory/AI instructions

#### `/docs` Directory (21 files - currently flat or poorly organized)
**User-Facing:**
- `CHANGELOG.md` - Version history
- `CLI_FEATURES.md` - CLI usage guide
- `CONFIGURATION.md` - Configuration reference
- `DEPLOY.md` - Installation/deployment
- `DEVELOPMENT.md` - Development workflow overview
- `FEATURES.md` - Feature overview
- `MCP_FILE_WATCHING.md` - File watching feature
- `mcp-integration.md` - MCP integration guide
- `RELEASES.md` - Release process
- `STRUCTURE.md` - Project structure
- `VERSIONING.md` - Versioning guide
- `VERSIONING_WORKFLOW.md` - Versioning workflow

**Developer-Focused:**
- `IMPROVEMENTS_SUMMARY.md` - Development summary

#### `/docs/developer/` (9 files)
- `API.md` - Internal API reference
- `CONTRIBUTING.md` - Contribution guidelines
- `DEVELOPER.md` - Technical architecture
- `LINTING.md` - Code quality/linting
- `REFACTORING_ANALYSIS.md` - Code analysis
- `TEST_SUITE_SUMMARY.md` - Test overview
- `TESTING_STRATEGY.md` - Testing approach
- `TESTING.md` - Testing guide

#### `/docs/reference/` (5 files)
- `ENGINEER_TASK.md` - Task documentation
- `INSTALL_COMMAND_ENHANCEMENTS.md` - Install command docs
- `INSTALL.md` - Installation guide
- `MCP_SETUP.md` - MCP setup guide
- `PROJECT_ORGANIZATION.md` - File organization standard

#### `/docs/architecture/` (1 file)
- `REINDEXING_WORKFLOW.md` - Reindexing architecture

#### `/docs/performance/` (2 files)
- `CONNECTION_POOLING.md` - Performance optimization
- `SEARCH_TIMING_ANALYSIS.md` - Search performance

#### `/docs/analysis/` (2 files)
- `SEARCH_ANALYSIS_REPORT.md` - Search analysis
- `SEARCH_IMPROVEMENT_PLAN.md` - Improvement plan

#### `/docs/debugging/` (1 file)
- `SEARCH_BUG_ANALYSIS.md` - Bug analysis

#### `/docs/technical/` (1 file)
- `SIMILARITY_CALCULATION_FIX.md` - Technical fix

#### `/docs/prd/` (1 file)
- `mcp_vector_search_prd_updated.md` - Product requirements (64KB - large)

#### `/docs/optimizations/` (1 file)
- `database-stats-chunked-processing.md` - Optimization doc

#### `/docs/_archive/` (3 files)
- Old versions of CLAUDE.md and summaries

---

## Issues Identified

### 1. **Structural Problems**
- Too many docs at root `/docs` level (21 files - overwhelming)
- Unclear separation between user and developer documentation
- Overlapping content (e.g., DEVELOPMENT.md, DEVELOPER.md, CONTRIBUTING.md)
- Inconsistent categorization (performance vs optimizations vs technical)
- Multiple "setup/install" docs scattered across locations

### 2. **Navigation Issues**
- No clear entry point for users vs developers
- Missing master index or table of contents
- Difficult to find specific information
- No clear learning path for new users

### 3. **Content Issues**
- Duplicate/overlapping content needs consolidation
- Some docs are outdated (e.g., ENGINEER_TASK.md)
- Inconsistent formatting and structure
- Missing cross-references between related docs

### 4. **Organization Issues**
- Inconsistent use of subdirectories
- Similar content in different categories
- No clear hierarchy or grouping logic

---

## Proposed New Structure

### Design Principles
1. **User Journey First**: Structure follows how users interact with the project
2. **Progressive Disclosure**: Basic → Intermediate → Advanced
3. **Clear Separation**: User docs vs Developer docs
4. **Single Source of Truth**: No duplicate content
5. **Easy Navigation**: Clear index, cross-references, breadcrumbs

### New Directory Structure

```
/docs/
├── index.md                          # NEW: Master documentation index
├── CHANGELOG.md                      # Keep at top level (users need this)
│
├── getting-started/                  # NEW: Onboarding
│   ├── README.md                    # NEW: Quick start guide
│   ├── installation.md              # CONSOLIDATED: INSTALL.md + DEPLOY.md
│   ├── first-steps.md               # NEW: Tutorial for new users
│   └── configuration.md             # MOVED: CONFIGURATION.md
│
├── guides/                           # NEW: User guides (how-to)
│   ├── README.md                    # NEW: Guide index
│   ├── cli-usage.md                 # MOVED: CLI_FEATURES.md
│   ├── mcp-integration.md           # CONSOLIDATED: mcp-integration.md + MCP_SETUP.md
│   ├── file-watching.md             # MOVED: MCP_FILE_WATCHING.md
│   ├── searching.md                 # NEW: Search guide extracted from README
│   └── indexing.md                  # NEW: Indexing guide extracted from README
│
├── reference/                        # API and technical reference
│   ├── README.md                    # NEW: Reference index
│   ├── cli-commands.md              # NEW: Complete CLI command reference
│   ├── configuration-options.md     # EXTRACTED: Detailed config from CONFIGURATION.md
│   ├── supported-languages.md       # NEW: Language support details
│   ├── architecture.md              # MOVED: STRUCTURE.md
│   └── features.md                  # MOVED: FEATURES.md
│
├── development/                      # Developer documentation
│   ├── README.md                    # NEW: Developer guide index
│   ├── setup.md                     # MOVED: DEVELOPMENT.md
│   ├── contributing.md              # MOVED: CONTRIBUTING.md
│   ├── architecture.md              # MOVED: DEVELOPER.md
│   ├── api.md                       # MOVED: API.md
│   ├── testing.md                   # CONSOLIDATED: TESTING.md + TESTING_STRATEGY.md + TEST_SUITE_SUMMARY.md
│   ├── code-quality.md              # MOVED: LINTING.md
│   ├── project-organization.md      # MOVED: PROJECT_ORGANIZATION.md
│   └── versioning.md                # CONSOLIDATED: VERSIONING.md + VERSIONING_WORKFLOW.md + RELEASES.md
│
├── architecture/                     # Architecture & design docs
│   ├── README.md                    # NEW: Architecture overview
│   ├── overview.md                  # NEW: High-level architecture
│   ├── indexing-workflow.md         # MOVED: REINDEXING_WORKFLOW.md
│   ├── performance.md               # CONSOLIDATED: CONNECTION_POOLING.md + performance docs
│   └── design-decisions.md          # NEW: ADR-style decisions
│
├── advanced/                         # NEW: Advanced topics
│   ├── README.md                    # NEW: Advanced topics index
│   ├── performance-tuning.md        # CONSOLIDATED: performance + optimizations
│   ├── embedding-models.md          # NEW: Model selection guide
│   ├── troubleshooting.md           # NEW: Common issues and solutions
│   └── extending.md                 # NEW: Adding language parsers, etc.
│
├── internal/                         # NEW: Internal/historical docs
│   ├── README.md                    # NEW: Internal docs note
│   ├── analysis-reports.md          # CONSOLIDATED: analysis/ directory
│   ├── improvements.md              # MOVED: IMPROVEMENTS_SUMMARY.md
│   ├── refactoring-analysis.md      # MOVED: REFACTORING_ANALYSIS.md
│   └── install-enhancements.md      # MOVED: INSTALL_COMMAND_ENHANCEMENTS.md
│
└── _archive/                         # Keep existing archive
    └── (existing archived files)
```

### Root Level Files (Unchanged)
```
/
├── README.md                         # Main entry point - keep as is
├── CLAUDE.md                         # AI/memory - keep as is
└── docs/                            # All documentation
```

---

## File Mapping

### Consolidations

#### Installation & Setup
**Target**: `docs/getting-started/installation.md`
**Sources**:
- `docs/DEPLOY.md`
- `docs/reference/INSTALL.md`
- `docs/reference/MCP_SETUP.md` (partial)

#### MCP Integration
**Target**: `docs/guides/mcp-integration.md`
**Sources**:
- `docs/mcp-integration.md`
- `docs/reference/MCP_SETUP.md`

#### Testing
**Target**: `docs/development/testing.md`
**Sources**:
- `docs/developer/TESTING.md`
- `docs/developer/TESTING_STRATEGY.md`
- `docs/developer/TEST_SUITE_SUMMARY.md`

#### Versioning & Releases
**Target**: `docs/development/versioning.md`
**Sources**:
- `docs/VERSIONING.md`
- `docs/VERSIONING_WORKFLOW.md`
- `docs/RELEASES.md`

#### Performance
**Target**: `docs/architecture/performance.md`
**Sources**:
- `docs/performance/CONNECTION_POOLING.md`
- `docs/performance/SEARCH_TIMING_ANALYSIS.md`
- `docs/optimizations/database-stats-chunked-processing.md`

#### Analysis & Reports
**Target**: `docs/internal/analysis-reports.md`
**Sources**:
- `docs/analysis/SEARCH_ANALYSIS_REPORT.md`
- `docs/analysis/SEARCH_IMPROVEMENT_PLAN.md`
- `docs/debugging/SEARCH_BUG_ANALYSIS.md`
- `docs/technical/SIMILARITY_CALCULATION_FIX.md`

### Direct Moves (Renamed)

| Old Path | New Path | Notes |
|----------|----------|-------|
| `docs/CONFIGURATION.md` | `docs/getting-started/configuration.md` | Move to getting started |
| `docs/CLI_FEATURES.md` | `docs/guides/cli-usage.md` | Rename for clarity |
| `docs/MCP_FILE_WATCHING.md` | `docs/guides/file-watching.md` | Better categorization |
| `docs/FEATURES.md` | `docs/reference/features.md` | Reference material |
| `docs/STRUCTURE.md` | `docs/reference/architecture.md` | Rename for clarity |
| `docs/DEVELOPMENT.md` | `docs/development/setup.md` | Move to development |
| `docs/developer/DEVELOPER.md` | `docs/development/architecture.md` | Consolidate dev docs |
| `docs/developer/API.md` | `docs/development/api.md` | Flatten structure |
| `docs/developer/CONTRIBUTING.md` | `docs/development/contributing.md` | Flatten structure |
| `docs/developer/LINTING.md` | `docs/development/code-quality.md` | Rename for clarity |
| `docs/reference/PROJECT_ORGANIZATION.md` | `docs/development/project-organization.md` | Dev-focused |
| `docs/architecture/REINDEXING_WORKFLOW.md` | `docs/architecture/indexing-workflow.md` | Rename |
| `docs/IMPROVEMENTS_SUMMARY.md` | `docs/internal/improvements.md` | Internal doc |
| `docs/developer/REFACTORING_ANALYSIS.md` | `docs/internal/refactoring-analysis.md` | Internal doc |
| `docs/reference/INSTALL_COMMAND_ENHANCEMENTS.md` | `docs/internal/install-enhancements.md` | Historical |

### Archive/Remove

| File | Action | Reason |
|------|--------|--------|
| `docs/reference/ENGINEER_TASK.md` | Archive | Likely outdated/historical |
| `docs/prd/mcp_vector_search_prd_updated.md` | Keep in prd/ | Product docs (separate from user docs) |

---

## New Files to Create

### Navigation & Indexes
1. `docs/index.md` - Master documentation index
2. `docs/getting-started/README.md` - Quick start guide
3. `docs/guides/README.md` - Guides overview
4. `docs/reference/README.md` - Reference overview
5. `docs/development/README.md` - Developer overview
6. `docs/architecture/README.md` - Architecture overview
7. `docs/advanced/README.md` - Advanced topics overview
8. `docs/internal/README.md` - Internal docs note

### New Content
1. `docs/getting-started/first-steps.md` - Tutorial for beginners
2. `docs/guides/searching.md` - Search usage guide
3. `docs/guides/indexing.md` - Indexing usage guide
4. `docs/reference/cli-commands.md` - Complete CLI reference
5. `docs/reference/configuration-options.md` - Detailed config options
6. `docs/reference/supported-languages.md` - Language support matrix
7. `docs/architecture/overview.md` - High-level architecture
8. `docs/architecture/design-decisions.md` - Architecture decisions
9. `docs/advanced/performance-tuning.md` - Performance guide
10. `docs/advanced/embedding-models.md` - Model selection
11. `docs/advanced/troubleshooting.md` - Common issues
12. `docs/advanced/extending.md` - Extensibility guide

---

## Migration Steps

### Phase 1: Create New Structure (No Breaking Changes)
1. Create all new directories
2. Create all README.md index files
3. Create new consolidated files
4. Leave old files in place

### Phase 2: Update Content
1. Consolidate duplicate content
2. Standardize formatting
3. Update cross-references
4. Add navigation elements

### Phase 3: Move Files
1. Git mv old files to new locations
2. Update all internal links
3. Update CLAUDE.md references
4. Update root README.md

### Phase 4: Verification
1. Check all links (automated)
2. Verify no broken references
3. Test navigation flow
4. Get feedback

### Phase 5: Cleanup
1. Remove old empty directories
2. Update .gitignore if needed
3. Commit with clear message

---

## Link Update Strategy

### Files Requiring Link Updates
1. `README.md` - Update all docs/ links
2. `CLAUDE.md` - Update reference links
3. All moved documentation files
4. Any scripts referencing docs

### Automated Link Checking
```bash
# Find all markdown links
grep -r "\[.*\](.*\.md)" docs/

# Validate links after reorganization
python scripts/validate-links.py
```

---

## Documentation Standards

### File Naming
- Use lowercase with hyphens: `cli-usage.md`
- README.md for directory indexes (uppercase)
- Descriptive names that indicate content

### Structure Standards
Every documentation file must have:
1. Clear H1 title
2. Brief description/purpose
3. Table of contents (for long docs)
4. Proper heading hierarchy (H1 → H2 → H3)
5. Code blocks with language hints
6. Cross-references to related docs

### Template for New Docs
```markdown
# Document Title

Brief description of what this document covers.

## Table of Contents
- [Section 1](#section-1)
- [Section 2](#section-2)

## Section 1
Content...

## Section 2
Content...

## Related Documentation
- [Link to related doc](./related.md)
```

---

## Success Criteria

✅ **Structure**
- Clear separation between user and developer documentation
- Logical grouping by purpose/audience
- Maximum 3 levels of nesting
- Each directory has a README.md index

✅ **Navigation**
- Master index at docs/index.md
- Clear breadcrumbs in each section
- Cross-references between related topics
- Easy to find information

✅ **Content**
- No duplicate information
- Consistent formatting
- All links working
- Up-to-date content

✅ **Discoverability**
- New users can get started quickly
- Developers can find technical details
- Advanced users can find performance tips
- Clear learning path

---

## Timeline

- **Phase 1**: Create structure (1-2 hours)
- **Phase 2**: Update content (2-3 hours)
- **Phase 3**: Move files (1 hour)
- **Phase 4**: Verification (1 hour)
- **Phase 5**: Cleanup (30 min)

**Total Estimated Time**: 5-7 hours

---

## Rollback Plan

If issues arise:
1. All changes are in git - easy revert
2. Old structure preserved during migration
3. Can pause at any phase
4. Incremental commits allow partial rollback

---

## Next Steps

1. ✅ Create this plan document
2. ⏳ Get approval/feedback
3. ⏳ Execute Phase 1: Create structure
4. ⏳ Execute Phase 2-5: Migrate content
5. ⏳ Final verification and commit
