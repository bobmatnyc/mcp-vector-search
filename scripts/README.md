# Scripts Directory

This directory contains various utility scripts for development, building, and deployment of the MCP Vector Search project.

## Active Scripts

### version_manager.py ⭐
**Main version management utility** - The primary tool for handling all version-related operations.

**Usage:**
```bash
python scripts/version_manager.py [command] [options]
```

**Key Commands:**
- `--show` - Display current version information
- `--bump [patch|minor|major]` - Increment version according to semantic versioning
- `--set VERSION --build NUMBER` - Set specific version and build
- `--increment-build` - Increment build number only
- `--update-changelog` - Add new version section to CHANGELOG.md
- `--git-commit` - Create git commit and tag for release
- `--dry-run` - Test operations without making changes

**Output Formats:**
- `--format simple` - Just the version number (e.g., "4.0.3")
- `--format detailed` - Full version information with build number
- `--format json` - Machine-readable JSON output

**Examples:**
```bash
# Show current version
python scripts/version_manager.py --show

# Bump patch version safely
python scripts/version_manager.py --bump patch --dry-run
python scripts/version_manager.py --bump patch

# Complete release workflow
python scripts/version_manager.py --bump minor --update-changelog --git-commit

# Set specific version
python scripts/version_manager.py --set 4.1.0 --build 300
```

**Features:**
- Semantic versioning compliance
- Automatic file updates (__init__.py, CHANGELOG.md)
- Git integration (commits, tags)
- Safe dry-run mode for testing
- Multiple output formats
- Comprehensive error handling

### Development & Testing Scripts

#### dev-test.sh
Development testing script for quick validation during development.

**Usage:**
```bash
./scripts/dev-test.sh [options]
```

**Features:**
- Quick test suite execution
- Development environment validation
- Code quality checks
- Local package testing

#### dev-setup.py
Development environment setup utility.

**Usage:**
```bash
python scripts/dev-setup.py [options]
```

**Features:**
- Development dependency installation
- Environment configuration
- Tool validation
- Project initialization

#### deploy-test.sh
Local deployment testing for validating distribution packages.

**Usage:**
```bash
./scripts/deploy-test.sh [options]
```

**Features:**
- Clean environment testing
- Package installation validation
- CLI functionality verification
- Distribution package testing

#### workflow.sh
Development workflow guidance and automation.

**Usage:**
```bash
./scripts/workflow.sh [command]
```

**Features:**
- Workflow step guidance
- Automated development processes
- Best practice enforcement
- Development stage management

### Performance & Analysis Scripts

#### benchmark_llm_models.py ⭐
Benchmarks various LLM models for the chat command to compare response quality, speed, token usage, and cost.

**Usage:**
```bash
# Test all models (via Makefile)
make benchmark-llm

# Test fast/cheap models only
make benchmark-llm-fast

# Test single query
make benchmark-llm-query QUERY="where is similarity_threshold configured?"

# Direct usage with custom options
uv run python scripts/benchmark_llm_models.py --models anthropic/claude-3-haiku
uv run python scripts/benchmark_llm_models.py --query "how does the indexer work?"
```

**Features:**
- Benchmarks 7 OpenRouter LLM models by default
- Tests response quality (★★★★★ rating)
- Measures latency (seconds)
- Tracks token usage (input/output)
- Calculates cost (USD per query)
- Provides recommendations (fastest, cheapest, best quality)
- Rate limiting protection (1s between requests)
- Comprehensive summary tables

**Models Tested:**
- Premium: `anthropic/claude-3.5-sonnet`, `openai/gpt-4o`
- Mid-tier: `anthropic/claude-3-haiku`, `openai/gpt-4o-mini`, `google/gemini-flash-1.5`
- Budget: `meta-llama/llama-3.1-70b-instruct`, `mistralai/mistral-large`

**Requirements:**
- OpenRouter API key: `export OPENROUTER_API_KEY='your-key'`
- Indexed project: Run `mcp-vector-search init && mcp-vector-search index`

**Documentation:** See [LLM Benchmarking Guide](../docs/guides/llm-benchmarking.md)

#### analyze_search_bottlenecks.py
Analyzes search performance and identifies bottlenecks in the search pipeline.

**Usage:**
```bash
python scripts/analyze_search_bottlenecks.py [options]
```

**Features:**
- Search latency profiling
- Database query analysis
- Embedding generation timing
- Memory usage tracking
- Performance recommendations

#### monitor_search_performance.py
Real-time monitoring of search performance metrics.

**Usage:**
```bash
python scripts/monitor_search_performance.py [options]
```

**Features:**
- Real-time performance monitoring
- Metrics collection and reporting
- Alert thresholds for performance degradation
- Historical performance tracking

#### quick_search_timing.py
Quick performance benchmarking for search operations.

**Usage:**
```bash
python scripts/quick_search_timing.py [query] [options]
```

**Features:**
- Single-query timing
- Multiple iteration benchmarks
- Performance comparison tools
- Simple output for CI integration

#### run_search_timing_tests.py
Comprehensive search timing test suite.

**Usage:**
```bash
python scripts/run_search_timing_tests.py [options]
```

**Features:**
- Automated test suite for search performance
- Multiple query types and patterns
- Statistical analysis of results
- Performance regression detection

### Testing Scripts

#### run_tests.py
Comprehensive test runner for the project.

**Usage:**
```bash
python scripts/run_tests.py [options]
```

**Features:**
- Full test suite execution
- Coverage reporting
- Test result analysis
- CI/CD integration support

#### test_connection_pooling.py
Specialized tests for connection pooling functionality.

**Usage:**
```bash
python scripts/test_connection_pooling.py [options]
```

**Features:**
- Connection pool performance testing
- Pool configuration validation
- Concurrent access testing
- Resource leak detection

#### test_reindexing_workflow.py
Testing for reindexing workflow and automation.

**Usage:**
```bash
python scripts/test_reindexing_workflow.py [options]
```

**Features:**
- Reindexing strategy validation
- Workflow testing
- Performance impact analysis
- Automation verification

### GitHub Project Management Scripts

#### setup_github_milestones.py
Automates creation of GitHub milestones and issue dependencies for project management.

**Usage:**
```bash
# Using GITHUB_TOKEN environment variable
export GITHUB_TOKEN="your_github_token_here"
python scripts/setup_github_milestones.py

# Using command-line argument
python scripts/setup_github_milestones.py --token "your_github_token_here"

# Dry run (test without making changes)
python scripts/setup_github_milestones.py --dry-run

# Skip milestone creation (only add dependencies)
python scripts/setup_github_milestones.py --skip-milestones

# Skip dependency updates (only create milestones)
python scripts/setup_github_milestones.py --skip-dependencies
```

**Features:**
- Automated milestone creation with due dates
- Issue assignment to milestones
- Dependency graph management
- Dry-run mode for testing
- Cross-platform compatibility

**Prerequisites:**
```bash
pip install PyGithub
```

**Documentation:** See [GitHub Milestones Setup Guide](../docs/development/github-milestones-setup.md)

#### setup_milestones.sh
Bash script alternative for creating GitHub milestones and assigning issues.

**Usage:**
```bash
./scripts/setup_milestones.sh
```

**Prerequisites:**
- GitHub CLI (`gh`) installed and authenticated
- `jq` for JSON parsing

#### add_issue_dependencies.sh
Adds dependency information to GitHub issue descriptions.

**Usage:**
```bash
./scripts/add_issue_dependencies.sh
```

**Features:**
- Automatic dependency section creation
- Blocked by / Blocks relationship tracking
- Idempotent operation (safe to run multiple times)

## Deprecated Scripts

⚠️ **These scripts are deprecated and should not be used.** Use the unified Makefile workflow instead.

### build.sh (DEPRECATED)
**Replaced by:** `make release-*` commands

Old build wrapper script that handled version bumping and package building.

**Migration:**
```bash
# Old way
./scripts/build.sh

# New way
make release-patch    # or release-minor, release-major
```

**Why deprecated:**
- Limited functionality compared to new Makefile system
- No dry-run support
- Inconsistent error handling
- Missing changelog integration

### dev-build.py (DEPRECATED)
**Replaced by:** `make version-*` commands and `scripts/version_manager.py`

Old version increment utility with limited functionality.

**Migration:**
```bash
# Old way
./scripts/dev-build.py --bump minor

# New way
make version-minor
# or directly:
python scripts/version_manager.py --bump minor
```

**Why deprecated:**
- Superseded by more comprehensive version_manager.py
- Limited integration with build system
- No changelog management
- Missing git integration

### publish.sh (DEPRECATED)
**Replaced by:** `make publish` command

Old publishing script for PyPI deployment.

**Migration:**
```bash
# Old way
./scripts/publish.sh

# New way
make publish          # for PyPI
make publish-test     # for TestPyPI
```

**Why deprecated:**
- No pre-flight checks
- Limited error handling
- Missing dry-run support
- No integration with version management

## Script Usage Guidelines

### Development Workflow
For most development tasks, use the Makefile commands rather than calling scripts directly:

```bash
# Version management
make version-show         # Instead of: python scripts/version_manager.py --show
make version-patch        # Instead of: python scripts/version_manager.py --bump patch

# Complete releases
make release-minor        # Full release workflow
make publish             # Publish to PyPI

# Development testing
make test                # Instead of: python scripts/run_tests.py
make dev                 # Instead of: ./scripts/dev-setup.py
```

### Direct Script Usage
Use scripts directly when you need:
- **Custom version operations** not covered by Makefile
- **Scripting and automation** in CI/CD pipelines
- **Advanced options** not exposed through Makefile
- **Integration** with other tools

### Performance Scripts Usage
Performance analysis scripts are used for:
- **Development optimization** - Finding bottlenecks during development
- **Regression testing** - Ensuring performance doesn't degrade
- **Production monitoring** - Tracking performance in deployed environments
- **Benchmarking** - Comparing performance across versions

Example workflow:
```bash
# 1. Baseline performance
python scripts/quick_search_timing.py "authentication"

# 2. Make performance changes
# ... code modifications ...

# 3. Re-test performance
python scripts/run_search_timing_tests.py --compare-baseline

# 4. Analyze detailed bottlenecks if needed
python scripts/analyze_search_bottlenecks.py --detailed
```

## Makefile Integration

Most scripts are integrated with the project Makefile for convenience:

| Script Operation | Makefile Command | Description |
|------------------|------------------|-------------|
| `version_manager.py --show` | `make version-show` | Display current version |
| `version_manager.py --bump patch` | `make version-patch` | Increment patch version |
| `version_manager.py --bump minor` | `make version-minor` | Increment minor version |
| `version_manager.py --bump major` | `make version-major` | Increment major version |
| Full release workflow | `make release-*` | Complete release process |
| `run_tests.py` | `make test` | Run test suite |
| `dev-setup.py` | `make dev` | Development setup |
| Performance testing | `make test-performance` | Run performance benchmarks |
| `benchmark_llm_models.py` | `make benchmark-llm` | Benchmark LLM models for chat |
| `benchmark_llm_models.py` (fast) | `make benchmark-llm-fast` | Benchmark fast/cheap models only |
| `benchmark_llm_models.py` (custom) | `make benchmark-llm-query QUERY="..."` | Benchmark single query |

See the [Versioning Workflow Documentation](../docs/VERSIONING_WORKFLOW.md) for complete details on the unified build and release system.

## Environment Requirements

All scripts require:
- **Python 3.11+** - Modern Python with type hints
- **UV package manager** - For dependency management
- **Git** - For version control operations

Performance scripts may require additional dependencies:
- **psutil** - For system monitoring
- **matplotlib** - For performance visualization
- **pandas** - For data analysis

Install development dependencies:
```bash
uv sync  # Install all dependencies including dev tools
```

## Adding New Scripts

When adding new scripts to this directory:

1. **Follow naming conventions**:
   - Use `snake_case.py` for Python scripts
   - Use descriptive names that indicate purpose
   - Add appropriate file extensions

2. **Include documentation**:
   - Add docstrings to all scripts
   - Include usage examples
   - Document command-line arguments

3. **Update this README**:
   - Add the script to the appropriate section
   - Include usage examples
   - Document key features

4. **Consider Makefile integration**:
   - Add frequently-used scripts to the Makefile
   - Provide convenient aliases for common operations
   - Maintain consistent interface patterns

5. **Follow project standards**:
   - Use project coding conventions
   - Include error handling
   - Support dry-run mode where applicable
   - Add comprehensive help text

## Troubleshooting

### Common Issues

#### Permission Errors
```bash
# Make scripts executable
chmod +x scripts/*.py
chmod +x scripts/*.sh
```

#### Python Path Issues
```bash
# Run from project root
cd /path/to/mcp-vector-search
python scripts/version_manager.py --show
```

#### Missing Dependencies
```bash
# Install development dependencies
uv sync

# Check specific requirements
python scripts/version_manager.py --help
```

#### Git Repository Issues
Scripts that interact with git require:
- Clean working directory for releases
- Proper git configuration (user.name, user.email)
- Remote repository access for pushing

```bash
# Check git status
git status

# Configure git if needed
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

#### Version Manager Issues
```bash
# Check current version state
python scripts/version_manager.py --show --format detailed

# Validate version file
cat src/mcp_vector_search/__init__.py | grep -E "(version|build)"

# Test version operations safely
python scripts/version_manager.py --bump patch --dry-run
```

### Debug Mode

Most scripts support verbose output:
```bash
# Version manager debug
python scripts/version_manager.py --show --format detailed

# Performance scripts with verbose output
python scripts/analyze_search_bottlenecks.py --verbose

# Test scripts with debug info
python scripts/run_tests.py --verbose
```

### Getting Help

Each script includes help documentation:
```bash
python scripts/version_manager.py --help
python scripts/analyze_search_bottlenecks.py --help
python scripts/run_tests.py --help
```

For Makefile operations:
```bash
make help  # Show all available commands
```

## Contributing

When modifying or adding scripts:

1. **Test thoroughly** - Use dry-run modes when available
2. **Update documentation** - Keep this README current
3. **Follow conventions** - Match existing code style
4. **Add error handling** - Scripts should fail gracefully
5. **Consider integration** - Should new functionality be added to Makefile?
6. **Write tests** - Add test coverage for new scripts
7. **Document changes** - Update CHANGELOG.md for significant changes

For more information, see the [Contributing Guidelines](../docs/developer/CONTRIBUTING.md).

## Script Categories Summary

### Version Management (Active)
- `version_manager.py` ⭐ - Primary version management tool

### Development (Active)
- `dev-test.sh` - Development testing
- `dev-setup.py` - Environment setup
- `deploy-test.sh` - Deployment testing
- `workflow.sh` - Workflow guidance

### Performance (Active)
- `benchmark_llm_models.py` ⭐ - LLM model benchmarking for chat
- `analyze_search_bottlenecks.py` - Performance analysis
- `monitor_search_performance.py` - Real-time monitoring
- `quick_search_timing.py` - Quick benchmarks
- `run_search_timing_tests.py` - Comprehensive timing tests

### Testing (Active)
- `run_tests.py` - Test runner
- `test_connection_pooling.py` - Connection pool tests
- `test_reindexing_workflow.py` - Reindexing tests

### GitHub Project Management (Active)
- `setup_github_milestones.py` - Automated milestone and dependency setup
- `setup_milestones.sh` - Bash alternative for milestone creation
- `add_issue_dependencies.sh` - Add dependency information to issues

### Deprecated (Do Not Use)
- `build.sh` - Use `make release-*` instead
- `dev-build.py` - Use `make version-*` instead
- `publish.sh` - Use `make publish` instead

**Recommendation:** Always prefer Makefile commands over direct script execution for standard operations.
