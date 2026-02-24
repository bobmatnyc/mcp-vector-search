# MCP Vector Search PR/MR Review Skill

## Overview
**Research-backed PR/MR review skill** optimized for maximum signal-to-noise ratio and developer productivity:

- **Evidence-based approach**: Built on industry research from Google, Microsoft, GitHub, and academic studies
- **Size-optimized reviews**: Automatic warnings for PRs >400 LOC (40% fewer defects when under limit)
- **Signal/noise filtering**: Tier-based feedback prioritization (>80% meaningful comments)
- **Cognitive load management**: Time-boxed review sessions to prevent fatigue
- **Context-aware analysis**: Vector search and knowledge graph for architectural reasoning
- **STAR feedback pattern**: Structured comments with Situation → Thinking → Alternative → Rationale
- **Comment labeling system**: Clear blocking vs. non-blocking distinction

## Usage
```
/mcp-vector-search-pr-review [--baseline=main] [--focus=security|performance|architecture]
/mcp-vector-search-pr-comments [PR_URL_OR_COMMENTS]
```

**Key Features:**

### 1. Branch Modification Review
- Only reviews files modified in your branch vs. baseline
- Uses `git diff` to identify exact changes
- Provides GitHub-style ```suggestion blocks for improvements
- Focuses on changed lines, not entire codebase

### 2. PR Comment Analysis & Response Planning
- Analyzes existing PR comments before addressing them
- Uses vector search to understand context of each comment
- Provides structured response plan with implementation guidance
- Prioritizes comments by impact and complexity

**Examples:**
```bash
# Review current branch vs main
/mcp-vector-search-pr-review --baseline=main

# Analyze PR comments and plan responses
/mcp-vector-search-pr-comments https://github.com/owner/repo/pull/123

# Paste PR comments for analysis
/mcp-vector-search-pr-comments "
Reviewer: Consider adding input validation in parse_json()
Reviewer: The error handling seems insufficient here
Reviewer: Can you add tests for the edge cases?
"

# Combined workflow: analyze comments, then review changes
/mcp-vector-search-pr-comments [PR_URL]
/mcp-vector-search-pr-review --baseline=main --focus=security
```

## 🎯 Research-Backed Framework

### Signal vs. Noise Prioritization

**Tier 1 — Critical Signal (Must Block)**
- `blocking:` Security vulnerabilities, correctness bugs, breaking API changes
- Race conditions, data loss risks, production failure scenarios
- **Target**: 100% of Tier 1 issues must be addressed before merge

**Tier 2 — Important Signal (Should Address)**
- `suggestion:` Architecture violations, missing error handling, performance issues
- API design decisions that are hard to reverse
- **Target**: >80% should be addressed, remainder can be follow-up tickets

**Tier 3 — Low Signal/Noise (Label as Non-blocking)**
- `nit:` Style preferences, naming, minor optimizations
- Alternative approaches that are equally valid
- **Target**: Author's discretion; should not block PR approval

### Comment Labeling System

**Use these labels for all feedback to eliminate approval ambiguity:**

| Label | Meaning | Example |
|-------|---------|---------|
| `blocking:` | Must be resolved before merge | `blocking: This SQL injection vulnerability needs input sanitization` |
| `suggestion:` | Worth considering, your call | `suggestion: Extract this validation logic into a separate function for reusability` |
| `nit:` | Minor style/preference; optional | `nit: Consider renaming \`d\` to \`document_count\` for clarity` |
| `question:` | Need to understand before approving | `question: Why use recursion here instead of iteration?` |
| `praise:` | Calling out excellent work | `praise: Excellent error handling with specific user-facing messages` |
| `thought:` | Sharing an idea; not requesting change | `thought: This pattern might be useful for the payment service too` |

### PR Size Warning System

**Automatic size analysis with research-backed thresholds:**

- **✅ Optimal (under 400 LOC)**: 40% fewer defects, 3x faster review cycles
- **⚠️ Large (400-800 LOC)**: Warning with suggestions for splitting
- **🚨 Too Large (over 800 LOC)**: Strong recommendation to break into smaller PRs
- **📊 Review time estimate**: +25 minutes per 100 additional lines

### STAR Feedback Pattern

**Structure all substantive feedback using:**

- **Situation**: Reference the specific code location
- **Thinking**: Explain your concern and reasoning
- **Alternative**: Suggest a better approach with code when helpful
- **Rationale**: Connect to why (performance, maintainability, correctness)

**Example:**
```markdown
suggestion: In `handle_review_repository()` at line 156, the current error handling only catches generic exceptions.

Consider adding specific exception types for different failure modes:
\```suggestion
try:
    result = await review_engine.run_review(...)
except ProjectNotFoundError as e:
    return error_response(f"Project not found: {e}")
except SearchTimeoutError as e:
    return error_response(f"Search timed out: {e}")
except Exception as e:
    logger.error("Unexpected error", exc_info=True)
    return error_response("Internal error occurred")
\```

This enables proper error recovery and debugging, plus gives users actionable error messages instead of generic failures.
```

## HOW-TO: Creating Effective PRs/MRs

### 1. Pre-Review Setup
```bash
# Ensure project is indexed for context
mvs index_project --force-reindex

# Build knowledge graph for dependency analysis
mvs kg_build

# Check current status
mvs kg_stats
```

### 2. PR Preparation Workflow
```bash
# 1. Create feature branch
git checkout -b feature/your-feature

# 2. After making changes, get contextual review
mvs analyze review-pr --baseline main --format github-json

# 3. Run focused analysis on changed files
mvs search_code "your changed functionality" --limit 10

# 4. Check for architectural impacts
mvs kg_query "functions calling YourChangedClass"

# 5. Validate with repository-wide security scan
mvs review_repository --review-type security --changed-only
```

## Review Process

### PR Comment Analysis (`/mcp-vector-search-pr-comments`)

**Research-optimized workflow for managing PR feedback:**

1. **Comment Parsing & Signal Classification**
   - Extract individual feedback items from PR comments
   - Classify each comment as Tier 1/2/3 using signal/noise framework
   - Identify `blocking:`, `suggestion:`, `nit:` labels (or infer from context)

2. **Context Discovery with Vector Search**
   - Use `search_code` to understand codebase context for each comment
   - Find similar patterns to validate or challenge reviewer suggestions
   - Discover existing implementations that solve similar problems

3. **Impact Assessment via Knowledge Graph**
   - Use `kg_query` to understand dependencies affected by suggested changes
   - Map downstream effects of each potential modification
   - Identify breaking changes that might not be obvious

4. **STAR-Structured Response Planning**
   - Generate implementation plan using Situation → Thinking → Alternative → Rationale
   - Provide code examples for complex suggestions
   - Estimate implementation effort and sequence dependencies

5. **Prioritization with Fatigue Prevention**
   - Rank by Tier 1 (blocking) → Tier 2 (important) → Tier 3 (optional)
   - Group related comments to minimize context switching
   - Time-box implementation sessions to prevent cognitive overload

### Branch Modification Review (`/mcp-vector-search-pr-review`)

**Size-optimized, signal-focused review workflow:**

1. **PR Size Assessment & Warning System**
   ```bash
   # Get PR metrics first
   lines_changed=$(git diff --stat origin/main...HEAD | tail -1 | awk '{print $4+$6}')
   if [ $lines_changed -gt 400 ]; then
     echo "⚠️ PR size: $lines_changed LOC (>400 optimal threshold)"
     echo "Consider splitting for 40% fewer defects and 3x faster reviews"
   fi
   ```

2. **Branch Analysis with Cognitive Load Limits**
   - Identify all modified files in the branch/PR
   - **Time-box review**: Maximum 60-90 minutes per session for optimal defect detection
   - **Break large reviews**: >800 LOC requires multiple focused sessions

3. **Context Discovery with Signal Filtering**
   - Use vector search to find similar patterns for changed code
   - Focus on architectural reasoning (humans) vs. style issues (automation)
   - Prioritize Tier 1/2 feedback that improves code health

4. **Impact Analysis via Knowledge Graph**
   - Query knowledge graph for affected downstream components
   - Identify cross-service dependencies that AI tools typically miss
   - Map potential breaking changes and rollback requirements

5. **STAR-Structured Suggestions**
   - Provide feedback using Situation → Thinking → Alternative → Rationale pattern
   - Use appropriate comment labels (`blocking:`, `suggestion:`, `nit:`)
   - Focus ONLY on changed lines and their immediate context

## Review Template: Branch Modification Focus

### Step 1: Changed Files Selection
```bash
# Get modified files only
git diff --name-only origin/main...HEAD
mvs review_pull_request --baseline main --changed-only
```

### Step 2: Context Discovery (Per Changed File)
```bash
# For each modified file, find related patterns
mvs search_code "similar functionality to YourChangedFile"
mvs kg_query "functions called by YourModifiedFunction"
```

### Step 3: STAR-Pattern Code Suggestions

**✅ CORRECT: Research-Backed Feedback with Labels**

**Example 1 - Critical Security Issue:**
```markdown
blocking: Empty LLM responses in `_parse_findings_json()` line 87 cause downstream crashes when the list is accessed.

The current code assumes non-empty responses, but LLM services can return empty strings during rate limiting or failures.

```suggestion
def _parse_findings_json(self, llm_response: str) -> list[ReviewFinding]:
    # Add input validation for empty/null responses
    if not llm_response or not llm_response.strip():
        logger.warning("Empty LLM response received")
        return []

    # Clean up common JSON issues before parsing
    json_str = self._clean_json_string(llm_response)
```

This prevents production crashes and provides proper logging for debugging rate limit issues.
```

**Example 2 - Architecture Improvement:**
```markdown
suggestion: The `handle_review_repository()` function at line 156 has generic exception handling that makes debugging difficult.

Current pattern catches all exceptions the same way, losing valuable error context for different failure modes.

```suggestion
async def handle_review_repository(self, args: dict[str, Any]) -> CallToolResult:
    try:
        # Validate review_type parameter early
        review_type_str = args.get("review_type", "security")
        if review_type_str not in ["security", "architecture", "performance"]:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Invalid review_type: {review_type_str}"
                )],
                isError=True,
            )
```

This enables proper error recovery and gives users actionable error messages instead of generic failures.
```

**Example 3 - Minor Style Preference:**
```markdown
nit: Variable name `d` at line 203 could be more descriptive.

```suggestion
# Consider renaming for clarity
document_count = len(processed_documents)  # instead of: d = len(processed_documents)
```

Not blocking, but would improve readability for future maintainers.
```

**❌ AVOID: Research-Identified Anti-Patterns**
- "Add validation in line 42" (no context or rationale)
- "Consider improving error handling" (vague, no specific suggestion)
- "This is wrong" (no explanation or alternative)
- Generic advice without code examples
- Blocking PRs on style preferences that should be automated
- Reviewing unchanged code outside the branch modifications

**✅ RESEARCH-BACKED BEST PRACTICES**
- Use comment labels (`blocking:`, `suggestion:`, `nit:`) for approval clarity
- Follow STAR pattern: Situation → Thinking → Alternative → Rationale
- Show exact code changes in ```suggestion blocks
- Focus ONLY on changed lines and their immediate context
- Prioritize Tier 1/2 feedback that measurably improves code health
- Provide concrete examples, not just descriptions

## PR Comment Response Planning

### Step 1: Comment Analysis & Context Discovery

For each PR comment, analyze using mcp-vector-search tools:

```bash
# For comment: "Add input validation in parse_json()"
mvs search_code "input validation json parsing" --limit 5
mvs search_code "validation patterns" --files "*.py"
mvs kg_query "functions calling parse_json"

# For comment: "Error handling seems insufficient"
mvs search_code "error handling patterns" --limit 5
mvs analyze_file src/handlers.py  # Get current error handling patterns
```

### Step 2: STAR-Pattern Response Plan Template

**Research-optimized comment response structure:**

```markdown
# PR Comment Response Plan

## Signal Classification Summary
- **Tier 1 (Blocking)**: 2 comments require resolution before merge
- **Tier 2 (Important)**: 1 comment should be addressed
- **Tier 3 (Optional)**: 1 nit-level suggestion
- **Estimated effort**: 3-4 hours across 2 focused sessions

---

## Comment 1: "Add input validation in parse_json()"
**Classification**: Tier 1 - `blocking:` (Security/reliability impact)

**Situation**: Current `parse_json()` function at line 156 accepts any input without validation
**Thinking**: Missing validation creates crash risk when LLM services return empty/malformed responses during rate limiting
**Alternative**: Add comprehensive input validation with specific error types
**Rationale**: Prevents production crashes and enables proper error recovery

**Context Discovery Results:**
- ✅ Similar validation in `validate_config()` at config.py:45 (pattern to follow)
- ✅ 3 other JSON parsing functions use try/except blocks (consistency)
- ⚠️ Knowledge graph shows 8 downstream callers affected by changes

**STAR-Structured Implementation:**
```suggestion
def parse_json(self, json_str: str) -> dict:
    # Situation: Add validation as requested by reviewer
    if not json_str or not json_str.strip():
        raise ValueError("JSON string cannot be empty")

    if not isinstance(json_str, str):
        raise TypeError(f"Expected string, got {type(json_str)}")

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Rationale: Specific error enables proper upstream handling
        raise ValueError(f"Invalid JSON: {e}") from e
```

**Dependencies & Testing:**
- Update: `handle_review_repository()`, `_parse_findings_json()`
- New tests: `test_parse_json_validation()`, `test_empty_json_handling()`

---

## Comment 2: "Error handling seems insufficient"
**Classification**: Tier 2 - `suggestion:` (Architecture improvement)

**Situation**: Generic exception handling in `handle_review_repository()` line 203
**Thinking**: Current catch-all pattern loses error context, making debugging difficult
**Alternative**: Specific exception types for different failure modes
**Rationale**: Enables proper error recovery and actionable user messages

**Context Discovery Results:**
- ✅ Similar handlers use `CallToolResult` pattern (follow existing convention)
- ✅ Knowledge graph identifies 8 downstream callers (impact scope)
- ✅ Found 3 similar error patterns in codebase (consistency opportunity)

**STAR-Structured Implementation:**
```suggestion
async def handle_review_repository(self, args: dict[str, Any]) -> CallToolResult:
    try:
        result = await review_engine.run_review(...)
        return CallToolResult(content=[...], isError=False)
    except ProjectNotFoundError as e:
        # Situation: User provided invalid project path
        logger.error(f"Project not found: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Project error: {e}")],
            isError=True
        )
    except SearchError as e:
        # Thinking: Search timeouts are recoverable, provide specific guidance
        logger.error(f"Search failed: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Search error: {e}")],
            isError=True
        )
    except Exception as e:
        # Rationale: Unexpected errors need full logging for debugging
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return CallToolResult(
            content=[TextContent(type="text", text="Internal error occurred")],
            isError=True
        )
```

---

## Comment 3: "Consider renaming variable `d`"
**Classification**: Tier 3 - `nit:` (Optional readability improvement)

**Situation**: Variable name `d` at line 245 represents document count
**Thinking**: Single-letter variables reduce code readability for future maintainers
**Alternative**: Descriptive name `document_count` follows project conventions
**Rationale**: Minor improvement, not blocking if author prefers current approach

**Implementation**: `d = len(docs)` → `document_count = len(docs)`
**Effort**: 30 seconds, author's discretion
```

### Step 3: Implementation Order & Cognitive Load Management

**Research-Optimized Sequence (Prevents Context Switching):**
1. **Tier 1 Blocking Issues** (Comment 1: Input validation)
   - No dependencies, enables safer error handling
   - Time estimate: 45-60 minutes
   - Session 1: Focus solely on validation logic

2. **Tier 2 Important Issues** (Comment 2: Error handling)
   - Builds on validation, affects all callers
   - Time estimate: 60-90 minutes
   - Session 2: After 15-minute break to restore cognitive capacity

3. **Tier 3 Optional Issues** (Comment 3: Naming improvements)
   - Quick wins during final review pass
   - Time estimate: 5-15 minutes

**Fatigue Prevention Protocol:**
- **Maximum 60-90 minutes continuous work** (optimal defect prevention)
- **15-minute breaks between sessions** to restore working memory
- **Group related changes** to minimize context switching overhead
- **Time-box each session** to prevent cognitive overload
- **Save complex architectural decisions** for when mental energy is highest

**Interdependency Management:**
- Input validation must come before error handling improvements
- Error handling changes affect all test scenarios
- Some comments may be resolved by implementing others (update response plan accordingly)

### Performance Focus Areas
- Vector database operations efficiency
- Knowledge graph query optimization
- Large file processing patterns
- Memory usage in indexing operations
- Concurrent processing bottlenecks

### Security Focus Areas
- Input validation in MCP tool handlers
- Path traversal prevention in file operations
- LLM response sanitization
- Configuration file security
- Dependency vulnerability checks

### Architecture Focus Areas
- Single responsibility in handler classes
- Proper error propagation patterns
- Clean separation between core/mcp layers
- Extension point design
- Testing strategy completeness

## Integration Points

### With mcp-vector-search Tools
- `review_pull_request` - For diff-based analysis
- `review_repository` - For comprehensive scans
- `search_code` - For pattern discovery
- `kg_query` - For impact analysis
- `analyze_file` - For quality metrics

### With Git Workflow
- Automatic baseline detection (main, develop, master)
- Changed file identification via git diff
- Commit message validation
- Branch naming convention checks

## Configuration

### Custom Review Instructions
Create `.mcp-vector-search/review-instructions.yaml`:
```yaml
language_standards:
  python:
    - Use type hints for all public methods
    - Follow PEP 8 naming conventions
    - Add docstrings for complex functions

scope_standards:
  mcp_handlers:
    - Validate all input parameters
    - Use proper error types (CallToolResult)
    - Include comprehensive logging

  core_modules:
    - Maintain backwards compatibility
    - Add comprehensive unit tests
    - Document performance characteristics

style_preferences:
  - Prefer composition over inheritance
  - Use dependency injection for testing
  - Minimize coupling between modules

custom_review_focus:
  - Vector search performance impacts
  - Knowledge graph consistency
  - MCP tool interface compliance
  - Error handling completeness
```

## Output Format: Branch-Focused Review

### Summary
- **Files Modified**: List of changed files in branch
- **Lines Changed**: Total additions/deletions
- **Context Analysis**: Similar patterns found via vector search
- **Impact Assessment**: Downstream effects via knowledge graph
- **Suggestions**: Count of actionable improvements for modified code only

### Branch Modification Review Format
```json
{
  "branch_summary": {
    "base_branch": "main",
    "modified_files": ["src/mcp/review_handlers.py", "tests/test_handlers.py"],
    "lines_added": 45,
    "lines_removed": 12,
    "overall_assessment": "APPROVE_WITH_SUGGESTIONS"
  },
  "file_suggestions": [
    {
      "file": "src/mcp/review_handlers.py",
      "modified_lines": "45-67",
      "suggestion_type": "performance",
      "priority": "medium",
      "github_suggestion": "```suggestion\n# Cache search_engine to avoid repeated initialization\nif not hasattr(self, '_search_engine_cache'):\n    embedding_function, _ = create_embedding_function(config.embedding_model)\n    self._search_engine_cache = SemanticSearchEngine(\n        database=database,\n        project_root=self.project_root\n    )\nreturn self._search_engine_cache\n```",
      "rationale": "Modified initialization code creates new search_engine on each call"
    }
  ],
  "context_analysis": {
    "similar_patterns_found": 3,
    "related_functions": ["create_database", "initialize_handlers"],
    "test_coverage_impact": "2 new test methods needed for modified functions"
  }
}
```

### Key Principles
- **Only Review Changed Code** - Don't suggest improvements to unmodified files
- **Use Git Context** - Compare against baseline branch (main/develop)
- **GitHub Suggestions** - All code improvements as ```suggestion blocks
- **Specific Over Generic** - "Add null check in line 45" not "improve validation"

## Research-Backed Best Practices

### For Reviewers: Maximum Signal, Minimum Noise

**🎯 Size-First Assessment (Most Critical Factor)**
```bash
# Check PR size BEFORE starting review
lines_changed=$(git diff --stat origin/main...HEAD | tail -1 | awk '{print $4+$6}')
if [ $lines_changed -gt 400 ]; then
  echo "⚠️ Large PR detected: $lines_changed LOC (Research: 40% more defects over 400 LOC)"
  echo "🔧 Suggest: Split into smaller PRs for better review quality"
fi
```

**⏱️ Cognitive Load Management**
1. **Time-box sessions**: 60-90 minutes maximum (research-proven defect detection peak)
2. **Take breaks**: 15-minute breaks every hour restore working memory capacity
3. **Batch similar work**: Group architecture reviews together, avoid context switching
4. **Review sequence**: Tests first (understand intent) → main logic → config/infrastructure

**🏷️ Signal Classification & Labeling**
- **Always** use comment labels: `blocking:`, `suggestion:`, `nit:`
- **Filter by tier**: Focus 80% of time on Tier 1/2 issues
- **STAR pattern**: Situation → Thinking → Alternative → Rationale for all substantive feedback
- **Avoid noise**: Don't block on style that should be automated

**🔍 Context Discovery Protocol**
1. **Start with git diff** - Always identify exact changed files/lines first
2. **Vector search per modification** - Run search for each changed function/class
3. **Knowledge graph impact** - Use KG to find downstream effects of changes
4. **Similar patterns** - Find existing implementations before suggesting alternatives

### Critical Review Commands (Research-Optimized)

**Phase 1: Size & Scope Assessment**
```bash
# Get PR metrics and warn if oversized
git diff --stat origin/main...HEAD
git diff --name-only origin/main...HEAD | wc -l  # File count
git diff --numstat origin/main...HEAD | awk '{added+=$1; deleted+=$2} END {print "Added:", added, "Deleted:", deleted}'
```

**Phase 2: Context-Aware Analysis**
```bash
# Review only the modified files (branch modification focus)
mvs review_pull_request --baseline main --format github-json
mvs review_repository --review-type security --changed-only

# Context discovery for each major change
mvs search_code "similar to YourModifiedClass" --limit 5
mvs kg_query "functions calling YourChangedMethod"
mvs analyze_file path/to/modified/file.py  # Quality metrics
```

**Phase 3: Signal-Focused Review**
```bash
# Tier 1 (blocking): Security and correctness
mvs review_repository --review-type security --changed-only
# Manual: Look for race conditions, null derefs, edge cases

# Tier 2 (important): Architecture and error handling
mvs review_repository --review-type architecture --changed-only
# Manual: API design, cross-service impact, error paths

# Tier 3 (optional): Code quality and style
mvs review_repository --review-type quality --changed-only
# Note: Style should be automated, focus on maintainability
```

### For Authors: Size-Optimized Development

**📏 Pre-PR Size Optimization (40% Fewer Defects)**
1. **Target <400 LOC per PR** - Research consensus for optimal review quality
2. **Vertical slicing** - Each PR delivers complete, independently deployable feature slice
3. **Separate concerns** - Refactoring-only PR → then feature PR on clean foundation
4. **Stacked PRs** - PR1 merged → PR2 merged → PR3 (use tools like Graphite for management)

**🔍 Pre-PR Self-Review Protocol**
```bash
# Size check first (most important quality factor)
git diff --stat origin/main...HEAD

# Self-review with signal focus
mvs review_pull_request --baseline main --changed-only
mvs search_code "YourNewPattern" --limit 5    # Check for existing solutions
mvs kg_query "impact of YourChanges"          # Validate dependencies
mvs analyze_file heavily_modified_file.py     # Performance check
```

**📋 PR Creation Checklist (Reduce Review Back-and-Forth)**
- [ ] PR size under 400 LOC (or documented reason for exception)
- [ ] Description explains **why** and **what alternatives were considered**
- [ ] Tests cover all modified functionality (not just happy path)
- [ ] Self-reviewed using vector search for similar patterns
- [ ] Breaking changes documented with migration path
- [ ] Performance impact assessed for hot paths

**🎯 Post-Feedback Response Protocol (Systematic + Efficient)**

**Phase 1: Strategic Analysis (Prevents Thrashing)**
```bash
# Analyze comments with signal classification
/mcp-vector-search-pr-comments [PR_URL_or_paste_comments]

# Understand codebase context BEFORE implementing
mvs search_code "validation patterns" --limit 5
mvs kg_query "functions calling parse_json"
```

**Phase 2: Prioritized Implementation (Prevents Fatigue)**
1. **Address Tier 1 first** - `blocking:` issues must be resolved
2. **Batch Tier 2 items** - `suggestion:` improvements in focused sessions
3. **Acknowledge Tier 3** - `nit:` items, author's discretion to implement
4. **Time-box work** - 60-90 minute focused sessions with breaks

**Phase 3: Verification Loop**
```bash
# Re-review changes after addressing feedback
mvs review_pull_request --baseline main --changed-only

# Verify no new issues introduced
mvs review_repository --review-type security --changed-only
```

**🔄 Complete Feedback Response Workflow**
```bash
# Step 1: Classify and analyze all feedback
/mcp-vector-search-pr-comments "$(pbpaste)"  # Paste PR comments

# Step 2: Context discovery (prevents misunderstanding reviewer intent)
mvs search_code "pattern mentioned by reviewer" --limit 5
mvs kg_query "dependencies affected by suggested change"

# Step 3: Implement in Tier order (prevents scope creep)
# Tier 1 (blocking) → break → Tier 2 (important) → break → Tier 3 (optional)

# Step 4: Verify and communicate
mvs review_pull_request --baseline main --changed-only
# Update PR with response summary and any decisions made
```

### Research-Identified Anti-Patterns to Avoid

**🚫 Reviewer Anti-Patterns (Create Noise, Reduce Signal)**
- ❌ **Review theater** - Approving without reading, meaningless feedback
- ❌ **Nitpick overload** - Blocking on style that should be automated (`nit:` outnumbers substantive feedback)
- ❌ **Moving goalposts** - New requirements each review round, expanding scope beyond PR
- ❌ **Vague feedback** - "This is confusing" without specific improvement suggestions
- ❌ **Context-free blocking** - "Add validation" without explaining why or how
- ❌ **Size blindness** - Attempting detailed review of 1000+ LOC PRs (research: 70% lower defect detection)

**🚫 Author Anti-Patterns (Reduce Review Quality)**
- ❌ **Mammoth PRs** - >800 LOC single PRs (research: 40% more defects, 3x slower reviews)
- ❌ **Missing context** - PR description just restates title, no ticket link, no alternatives discussed
- ❌ **Pre-review submission** - Not running tests locally, submitting without self-review
- ❌ **Defensive responses** - Treating review as personal criticism vs. collaborative improvement
- ❌ **Scope expansion** - Adding "one more thing" during review cycle

**🚫 Team Anti-Patterns (System Failures)**
- ❌ **No automation boundary** - Humans reviewing style/formatting instead of focusing on architecture
- ❌ **Review SLA vacuum** - No defined turnaround time, PRs blocking indefinitely
- ❌ **Blocking ambiguity** - Authors unclear what's required vs. optional for merge
- ❌ **Review isolation** - No design review before implementation (architecture discussions too late)
- ❌ **Context switching chaos** - No batched review windows, constant interruption-driven review

---

**✅ Research-Backed Success Patterns**

**📏 Size Management**
- Keep PRs under 400 LOC (40% fewer defects, 3x faster reviews)
- Use stacked PRs and vertical slicing for large features
- Separate refactoring from feature changes

**🏷️ Signal Optimization**
- Use comment labels for approval clarity (`blocking:`, `suggestion:`, `nit:`)
- Apply STAR feedback pattern for substantive comments
- Focus 80% of review time on Tier 1/2 issues

**🧠 Cognitive Load Management**
- Time-box review sessions to 60-90 minutes maximum
- Take 15-minute breaks between sessions
- Batch similar review work to minimize context switching

**🔍 Context-Aware Analysis**
- Review only changed lines and immediate context
- Use vector search and KG for architectural reasoning
- Provide specific code suggestions, not just descriptions

**⚡ Efficient Workflows**
- Separate automation (style, basic security) from human judgment
- Establish clear blocking vs. non-blocking standards
- Create review batching windows to prevent constant interruption

## Installation

This skill is automatically installed with mcp-vector-search and available in any project with `.mcp-vector-search/` configuration.

```bash
pip install mcp-vector-search[skills]
# Skill available as: /mcp-vector-search-pr-review
```

---

## Research Foundation

This skill is built on comprehensive analysis of industry best practices from:
- **Google Engineering Practices** - Small CLs, readability standards, review velocity
- **Microsoft/GitHub Research** - 1.5M+ review comments analysis, PR size impact studies
- **Academic Studies** - Cisco/SmartBear (2,500+ reviews), LinearB (6.1M PRs), cognitive load research
- **Industry Data** - 40% fewer defects under 400 LOC, 3x faster review cycles, 80% noise reduction targets

**Evidence-Based Features:**
- Signal/noise framework with >80% meaningful feedback target
- PR size warnings based on defect correlation research
- STAR feedback pattern for constructive code review
- Cognitive load management with time-boxed sessions
- Comment labeling system eliminating approval ambiguity

---
**Version**: 2.0.0 (Research-Backed)
**Compatible**: mcp-vector-search 3.0.15+
**Research Update**: 2026-02-23
**Author**: MCP Vector Search Team
