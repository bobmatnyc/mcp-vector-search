# Test Quality Examples - Good vs Bad Patterns

**Date:** 2026-01-12
**Purpose:** Real examples from mcp-vector-search codebase showing good and bad test patterns

## ✅ Good Test Patterns

### 1. Well-Structured Unit Test

**File:** `tests/unit/test_mcp_install_auto_detection.py`

**Why it's good:**
- Clear test class organization
- Descriptive test names
- One assertion per logical concept
- Good use of pytest fixtures
- Tests behavior, not implementation

```python
class TestAutoDetectProjectRoot:
    """Test automatic project root detection."""

    def test_detect_with_mcp_vector_search_directory(self, tmp_path: Path):
        """Should detect project root via .mcp-vector-search directory."""
        # Arrange
        (tmp_path / ".mcp-vector-search").mkdir()

        # Act
        result = auto_detect_project_root(tmp_path)

        # Assert
        assert result == tmp_path

    def test_detect_fallback_to_current_directory(self, tmp_path: Path):
        """Should fallback to current directory if no markers found."""
        # Arrange - No .git or .mcp-vector-search

        # Act
        result = auto_detect_project_root(tmp_path)

        # Assert
        assert result == tmp_path
```

**Good practices:**
- ✅ Arrange-Act-Assert pattern
- ✅ Descriptive docstrings
- ✅ Single responsibility per test
- ✅ Clear test data setup

### 2. Good Fixture Design

**File:** `tests/conftest.py`

**Why it's good:**
- Reusable across tests
- Well-documented
- Proper cleanup
- Returns useful test data

```python
@pytest.fixture
def temp_project_dir(temp_dir: Path) -> Path:
    """Create a temporary project directory with sample files."""
    project_dir = temp_dir / "test_project"
    project_dir.mkdir()

    # Create sample Python files
    sample_files = {
        "main.py": '''
def main():
    """Main application entry point."""
    print("Hello, World!")
    user_service = UserService()
    users = user_service.get_all_users()
    return len(users)
''',
        # ... more sample files
    }

    for filename, content in sample_files.items():
        (project_dir / filename).write_text(content)

    return project_dir
```

**Good practices:**
- ✅ Clear fixture purpose
- ✅ Returns usable test data
- ✅ Automatic cleanup (temp_dir context manager)
- ✅ Realistic sample data

### 3. Good Assertion Helper

**File:** `tests/conftest.py`

**Why it's good:**
- Reusable validation logic
- Clear error messages
- Validates multiple properties

```python
def assert_search_results_valid(results: list[SearchResult], min_count: int = 0):
    """Assert that search results are valid."""
    assert len(results) >= min_count, (
        f"Expected at least {min_count} results, got {len(results)}"
    )

    for result in results:
        assert isinstance(result, SearchResult)
        assert result.content is not None
        assert 0.0 <= result.similarity_score <= 1.0
        assert result.file_path is not None
        assert result.start_line >= 0
        assert result.end_line >= result.start_line
```

**Good practices:**
- ✅ Descriptive error messages
- ✅ Validates all important properties
- ✅ Reusable across test files

### 4. Good Integration Test

**File:** `tests/integration/test_indexing_workflow.py`

**Why it's good:**
- Tests complete workflow
- Uses real components (not mocked)
- Verifies end-to-end behavior

```python
@pytest.mark.asyncio
async def test_complete_indexing_workflow(temp_project_dir):
    """Test complete indexing workflow from initialization to search."""
    # Initialize project
    project_manager = ProjectManager(temp_project_dir)
    config = project_manager.initialize(
        file_extensions=[".py"],
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )

    # Create components
    embedding_function, _ = create_embedding_function(config.embedding_model)
    database = ChromaVectorDatabase(
        persist_directory=config.index_path,
        embedding_function=embedding_function,
    )

    # Index project
    async with database:
        indexer = SemanticIndexer(database, temp_project_dir, [".py"])
        indexed_count = await indexer.index_project()

        assert indexed_count > 0

        # Verify search works
        search_engine = SemanticSearchEngine(database, temp_project_dir)
        results = await search_engine.search("function", limit=5)

        assert len(results) > 0
        assert_search_results_valid(results)
```

**Good practices:**
- ✅ Tests real integration, not mocks
- ✅ Verifies complete workflow
- ✅ Clear test progression
- ✅ Meaningful assertions

## ❌ Bad Test Patterns

### 1. Skipped Test Dead Code

**File:** `tests/unit/commands/test_setup.py` (lines 238-914)

**Why it's bad:**
- 676 lines of code that never runs
- Maintenance burden for no value
- Misleading - looks like tests exist but they're skipped

```python
@pytest.mark.skip(
    reason="Tests need update for py_mcp_installer PlatformInfo data structure"
)
class TestSetupCommand:
    """Test suite for main setup command functionality."""

    @pytest.mark.asyncio
    async def test_setup_fresh_project(self, mock_python_project, mock_typer_context):
        """Test complete setup in fresh project."""
        # ... 40 lines of dead code ...
        pass

    # ... 10 more tests, all dead code ...
```

**Problems:**
- ❌ Skipped for months without resolution
- ❌ 74% of file is dead code
- ❌ No plan to fix or remove
- ❌ Blocks future development

**Fix:** DELETE or UPDATE immediately

### 2. Empty Test Placeholders

**File:** `tests/unit/cli/test_chat_analyze.py` (lines 313-368)

**Why it's bad:**
- Tests that don't test anything
- Give false confidence
- Waste reviewer time

```python
def test_analyze_markdown_files(self):
    pass

def test_analyze_javascript_project(self):
    pass

def test_analyze_typescript_project(self):
    pass

def test_analyze_rust_project(self):
    pass

# ... 6 more similar empty tests
```

**Problems:**
- ❌ No actual testing happening
- ❌ Tests pass but provide zero value
- ❌ False test count inflation (10 "tests")
- ❌ Placeholder been there for months

**Fix:** DELETE or IMPLEMENT properly

### 3. Duplicate Test Coverage

**Files:** `tests/test_simple.py` vs `tests/test_basic_functionality.py`

**Why it's bad:**
- Wastes CI time running same tests twice
- Maintenance burden (update both)
- Confusing - which one is canonical?

```python
# test_simple.py (253 lines)
@pytest.mark.asyncio
async def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    components = await ComponentFactory.create_standard_components(
        project_root=project_dir,
        use_pooling=False,
        include_search_engine=False,
        include_auto_indexer=False,
    )
    assert components.project_manager is not None
    # ... more assertions ...

# test_basic_functionality.py (168 lines)
def test_project_initialization(temp_project_dir):
    """Test project initialization."""
    project_manager = ProjectManager(temp_project_dir)
    assert not project_manager.is_initialized()
    # ... similar coverage ...
```

**Problems:**
- ❌ Two files testing the same functionality
- ❌ Different approaches to same test
- ❌ Wastes ~2 seconds of CI time
- ❌ 253 lines of redundant code

**Fix:** DELETE test_simple.py, keep test_basic_functionality.py

### 4. Integration Test Disguised as Unit Test

**File:** `tests/unit/cli/test_chat_analyze.py`

**Why it's bad:**
- In unit test directory but requires 6+ mocks
- Skipped because mocking too complex
- Should be integration test

```python
@pytest.mark.skip(reason="Integration test - requires full mocking")
@pytest.mark.asyncio
async def test_analyze_basic_query(self, mock_project_root, mock_llm_client):
    """Test basic analysis query."""
    with (
        patch("mcp_vector_search.core.llm_client.LLMClient") as mock_llm_client,
        patch("mcp_vector_search.core.project.ProjectManager") as mock_project_manager,
        patch("mcp_vector_search.core.config_utils.get_openai_api_key") as mock_openai_key,
        patch("mcp_vector_search.core.config_utils.get_openrouter_api_key") as mock_openrouter_key,
        patch("mcp_vector_search.parsers.registry.ParserRegistry") as mock_parser_registry,
        patch("mcp_vector_search.analysis.ProjectMetrics") as mock_project_metrics,
        patch("mcp_vector_search.analysis.interpretation.EnhancedJSONExporter") as mock_exporter,
    ):
        # ... complex mock setup ...
        pass
```

**Problems:**
- ❌ 6+ mocks = not a unit test
- ❌ In wrong directory (unit/ vs integration/)
- ❌ Mocking too complex, so skipped
- ❌ Misleading test organization

**Fix:** MOVE to tests/integration/ or refactor as true unit test

### 5. Test with Main Function

**File:** `tests/test_simple.py`

**Why it's bad:**
- Tests should be run by pytest, not have main()
- Mixes test code with CLI entry points
- Not a proper test structure

```python
async def main():
    """Test basic functionality."""
    print("🚀 Testing MCP Vector Search...")

    with tempfile.TemporaryDirectory() as temp_dir:
        project_dir = Path(temp_dir)
        # ... 140 lines of test logic ...

    print("\n🎉 Test completed!")


def main_wrapper():
    """Main function wrapper for CLI usage."""
    try:
        basic_result = asyncio.run(test_basic_functionality())
        if not basic_result:
            print("💥 Basic functionality test failed!")
            sys.exit(1)

        asyncio.run(main())
        print("🎉 All tests completed!")
        sys.exit(0)
    except Exception as e:
        print(f"💥 Test error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main_wrapper()
```

**Problems:**
- ❌ main() in test file (should be pytest-runnable)
- ❌ Manual exit codes instead of pytest assertions
- ❌ Print statements instead of proper test output
- ❌ 140 lines of duplicate test logic

**Fix:** DELETE - use pytest properly

### 6. Overly Large Test File

**File:** `tests/unit/analysis/visualizer/test_d3_visualization.py` (1,131 lines)

**Why it's bad:**
- Too large to review effectively
- Multiple concerns mixed together
- Hard to navigate
- Slow to run all tests

```python
# test_d3_visualization.py - 1,131 lines!

class TestD3LayoutAlgorithm:
    # ... 200 lines ...

class TestD3Interactions:
    # ... 250 lines ...

class TestD3Rendering:
    # ... 300 lines ...

class TestD3DataBinding:
    # ... 150 lines ...

class TestD3EdgeCases:
    # ... 200 lines ...
```

**Problems:**
- ❌ 1,131 lines in single file
- ❌ Multiple test classes for different concerns
- ❌ Hard to find specific test
- ❌ Takes 5+ seconds to run all tests in file

**Fix:** SPLIT into 4 focused test files

### 7. Test Without Meaningful Assertions

**Anti-pattern example** (not from codebase, but watch for this):

```python
def test_parse_python_file(self):
    """Test that Python file parsing doesn't crash."""
    try:
        result = parser.parse_file("test.py")
        # No assertions - just checking it doesn't crash!
        pass
    except Exception:
        pass  # Even worse - swallowing exceptions!
```

**Problems:**
- ❌ No assertions about result
- ❌ Test passes even if result is wrong
- ❌ Just checking "doesn't crash"
- ❌ Swallowing exceptions

**Fix:**
```python
def test_parse_python_file(self):
    """Test that Python file parsing returns valid AST."""
    result = parser.parse_file("test.py")

    # Meaningful assertions
    assert result is not None
    assert len(result.functions) > 0
    assert result.functions[0].name == "expected_function_name"
    assert result.language == "python"
```

## Test Organization Patterns

### ✅ Good Organization

```
tests/
├── unit/                      # Pure unit tests
│   ├── analysis/             # Grouped by module
│   │   ├── test_metrics.py  # Focused files
│   │   ├── test_smells.py   # <500 lines each
│   │   └── test_debt.py
│   ├── cli/
│   │   ├── test_status.py
│   │   └── test_search_quality_filters.py
│   └── core/
│       ├── test_database.py
│       └── test_search.py
├── integration/              # Multi-component tests
│   ├── test_indexing_workflow.py
│   └── test_boilerplate_filtering.py
├── e2e/                     # End-to-end tests
│   └── test_cli_commands.py
└── conftest.py              # Shared fixtures
```

### ❌ Bad Organization

```
tests/
├── test_simple.py           # Duplicate of test_basic_functionality.py
├── test_basic_functionality.py  # Should be in unit/
├── test_dart_parser.py      # Should be in unit/parsers/
├── test_html_parser.py      # Should be in unit/parsers/
├── test_js_parser.py        # Should be in unit/parsers/
├── test_php_parser.py       # Should be in unit/parsers/
├── test_ruby_parser.py      # Should be in unit/parsers/
├── test_search_performance.py  # Should be in performance/
├── manual/                  # 70 files (39 are not tests!)
│   ├── test_*.py (31 files)
│   ├── *.html (8 files)    # Debug artifacts
│   ├── *.png (2 files)      # Screenshots
│   ├── *.md (5 files)       # Notes
│   └── *.sh (5 files)       # Scripts
└── unit/
    └── cli/
        └── test_chat_analyze.py  # Integration test in unit directory!
```

## Checklist for Writing Good Tests

### Before Writing
- [ ] Understand what behavior you're testing
- [ ] Choose correct test type (unit/integration/e2e)
- [ ] Place in appropriate directory
- [ ] Check if test already exists

### While Writing
- [ ] Use descriptive test name (test_what_when_then)
- [ ] Write clear docstring
- [ ] Follow Arrange-Act-Assert pattern
- [ ] One logical assertion per test
- [ ] Use appropriate fixtures
- [ ] Avoid unnecessary mocks (prefer real objects)

### After Writing
- [ ] Test passes reliably
- [ ] Test fails when it should (verify negative case)
- [ ] Clear error messages when fails
- [ ] No print statements (use assertions)
- [ ] No commented-out code
- [ ] Follows project conventions

## Red Flags to Watch For

🚩 **Test has `@pytest.mark.skip` for more than 1 week**
- Decide: Fix or delete

🚩 **Test body is just `pass`**
- Decide: Implement or delete

🚩 **Test has 4+ mock.patch() calls**
- This is an integration test, move to tests/integration/

🚩 **Test file is >800 lines**
- Split into multiple focused files

🚩 **Test has no assertions**
- Add meaningful assertions or delete

🚩 **Test has try/except with pass**
- Remove - let test fail naturally

🚩 **Test has main() or if __name__ == "__main__"**
- Tests should be pytest-runnable only

🚩 **Test duplicates another test**
- Delete duplicate, keep best version

🚩 **Test in wrong directory** (unit vs integration vs e2e)
- Move to correct location

🚩 **Test file has "simple" or "basic" in name**
- Usually a duplicate, review and consolidate

## Summary

**Good tests are:**
- ✅ Focused (one behavior)
- ✅ Fast (unit tests especially)
- ✅ Independent (no test order dependency)
- ✅ Repeatable (same result every time)
- ✅ Self-validating (clear pass/fail)
- ✅ Timely (written close to code)

**Bad tests are:**
- ❌ Skipped indefinitely
- ❌ Empty placeholders
- ❌ Duplicate coverage
- ❌ In wrong directory
- ❌ Over-mocked
- ❌ No assertions
- ❌ Too large (>800 lines)

**Remember:**
- Tests are documentation of behavior
- Skipped tests are technical debt
- Empty tests are worse than no tests
- Good organization enables maintainability
