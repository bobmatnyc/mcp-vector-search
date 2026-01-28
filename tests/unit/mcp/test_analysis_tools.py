"""Unit tests for MCP analysis tools."""

import json
from pathlib import Path
from typing import Any

import pytest

from mcp_vector_search.mcp.server import MCPVectorSearchServer


@pytest.fixture
def mock_project_root(tmp_path: Path) -> Path:
    """Create a temporary project root."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()

    # Create sample Python file
    sample_file = project_dir / "sample.py"
    sample_file.write_text(
        """
def simple_function(x: int) -> int:
    '''A simple function.'''
    return x * 2

class SampleClass:
    '''A sample class.'''

    def method_one(self, a: int, b: int) -> int:
        '''Method with some complexity.'''
        if a > 0:
            if b > 0:
                return a + b
            else:
                return a
        else:
            return 0
"""
    )

    return project_dir


@pytest.fixture
def mcp_server(mock_project_root: Path) -> MCPVectorSearchServer:
    """Create an MCP server instance."""
    return MCPVectorSearchServer(project_root=mock_project_root)


@pytest.mark.asyncio
@pytest.mark.skip(
    reason="Obsolete tests after v1.2.7 refactoring - MCP server handlers extracted to separate modules"
)
class TestAnalyzeProject:
    """Tests for analyze_project tool."""

    async def test_analyze_project_summary_format(
        self, mcp_server: MCPVectorSearchServer
    ) -> None:
        """Test analyze_project with summary output format."""
        args = {"threshold_preset": "standard", "output_format": "summary"}

        result = await mcp_server._analyze_project(args)

        assert not result.isError
        assert len(result.content) == 1
        response_text = result.content[0].text

        # Verify summary contains expected sections
        assert "Project Analysis Summary" in response_text
        assert "Total Files:" in response_text
        assert "Complexity Distribution" in response_text
        assert "Health Metrics" in response_text
        assert "Code Smells" in response_text

    async def test_analyze_project_detailed_format(
        self, mcp_server: MCPVectorSearchServer
    ) -> None:
        """Test analyze_project with detailed JSON output format."""
        args = {"threshold_preset": "standard", "output_format": "detailed"}

        result = await mcp_server._analyze_project(args)

        assert not result.isError
        assert len(result.content) == 1
        response_text = result.content[0].text

        # Should be valid JSON
        output = json.loads(response_text)
        assert "project_root" in output
        assert "total_files" in output
        assert "total_functions" in output
        assert "complexity_distribution" in output
        assert "smells" in output

    async def test_analyze_project_strict_preset(
        self, mcp_server: MCPVectorSearchServer
    ) -> None:
        """Test analyze_project with strict threshold preset."""
        args = {"threshold_preset": "strict", "output_format": "summary"}

        result = await mcp_server._analyze_project(args)

        assert not result.isError
        # Strict preset should potentially detect more smells
        assert len(result.content) == 1

    async def test_analyze_project_relaxed_preset(
        self, mcp_server: MCPVectorSearchServer
    ) -> None:
        """Test analyze_project with relaxed threshold preset."""
        args = {"threshold_preset": "relaxed", "output_format": "summary"}

        result = await mcp_server._analyze_project(args)

        assert not result.isError
        assert len(result.content) == 1

    async def test_analyze_project_no_files(self, tmp_path: Path) -> None:
        """Test analyze_project with no analyzable files."""
        empty_server = MCPVectorSearchServer(project_root=tmp_path)
        args = {"threshold_preset": "standard", "output_format": "summary"}

        result = await empty_server._analyze_project(args)

        assert result.isError
        assert "No analyzable files found" in result.content[0].text


@pytest.mark.asyncio
@pytest.mark.skip(
    reason="Obsolete tests after v1.2.7 refactoring - MCP server handlers extracted to separate modules"
)
class TestAnalyzeFile:
    """Tests for analyze_file tool."""

    async def test_analyze_file_success(
        self, mcp_server: MCPVectorSearchServer, mock_project_root: Path
    ) -> None:
        """Test successful file analysis."""
        file_path = "sample.py"
        args = {"file_path": file_path}

        result = await mcp_server._analyze_file(args)

        assert not result.isError
        assert len(result.content) == 1
        response_text = result.content[0].text

        # Verify file analysis contains expected sections
        assert "File Analysis:" in response_text
        assert "Total Lines:" in response_text
        assert "Complexity Metrics" in response_text
        assert "Code Smells" in response_text

    async def test_analyze_file_absolute_path(
        self, mcp_server: MCPVectorSearchServer, mock_project_root: Path
    ) -> None:
        """Test file analysis with absolute path."""
        file_path = str(mock_project_root / "sample.py")
        args = {"file_path": file_path}

        result = await mcp_server._analyze_file(args)

        assert not result.isError
        assert len(result.content) == 1

    async def test_analyze_file_not_found(
        self, mcp_server: MCPVectorSearchServer
    ) -> None:
        """Test file analysis with non-existent file."""
        args = {"file_path": "nonexistent.py"}

        result = await mcp_server._analyze_file(args)

        assert result.isError
        assert "File not found" in result.content[0].text

    async def test_analyze_file_missing_parameter(
        self, mcp_server: MCPVectorSearchServer
    ) -> None:
        """Test file analysis with missing file_path parameter."""
        args: dict[str, Any] = {}

        result = await mcp_server._analyze_file(args)

        assert result.isError
        assert "file_path parameter is required" in result.content[0].text


@pytest.mark.asyncio
@pytest.mark.skip(
    reason="Obsolete tests after v1.2.7 refactoring - MCP server handlers extracted to separate modules"
)
class TestFindSmells:
    """Tests for find_smells tool."""

    async def test_find_smells_all(self, mcp_server: MCPVectorSearchServer) -> None:
        """Test finding all code smells."""
        args: dict[str, Any] = {}

        result = await mcp_server._find_smells(args)

        assert not result.isError
        assert len(result.content) == 1
        # Should contain smells or "No code smells found"
        response_text = result.content[0].text
        assert (
            "Code Smells Found" in response_text
            or "No code smells found" in response_text
        )

    async def test_find_smells_by_type(self, mcp_server: MCPVectorSearchServer) -> None:
        """Test finding smells filtered by type."""
        args = {"smell_type": "Long Method"}

        result = await mcp_server._find_smells(args)

        assert not result.isError
        assert len(result.content) == 1

    async def test_find_smells_by_severity(
        self, mcp_server: MCPVectorSearchServer
    ) -> None:
        """Test finding smells filtered by severity."""
        args = {"severity": "warning"}

        result = await mcp_server._find_smells(args)

        assert not result.isError
        assert len(result.content) == 1

    async def test_find_smells_combined_filters(
        self, mcp_server: MCPVectorSearchServer
    ) -> None:
        """Test finding smells with combined filters."""
        args = {"smell_type": "Complex Method", "severity": "error"}

        result = await mcp_server._find_smells(args)

        assert not result.isError
        assert len(result.content) == 1


@pytest.mark.asyncio
@pytest.mark.skip(
    reason="Obsolete tests after v1.2.7 refactoring - MCP server handlers extracted to separate modules"
)
class TestGetComplexityHotspots:
    """Tests for get_complexity_hotspots tool."""

    async def test_get_hotspots_default_limit(
        self, mcp_server: MCPVectorSearchServer
    ) -> None:
        """Test getting complexity hotspots with default limit."""
        args: dict[str, Any] = {}

        result = await mcp_server._get_complexity_hotspots(args)

        assert not result.isError
        assert len(result.content) == 1
        response_text = result.content[0].text
        assert (
            "Complexity Hotspots" in response_text
            or "No complexity hotspots found" in response_text
        )

    async def test_get_hotspots_custom_limit(
        self, mcp_server: MCPVectorSearchServer
    ) -> None:
        """Test getting complexity hotspots with custom limit."""
        args = {"limit": 5}

        result = await mcp_server._get_complexity_hotspots(args)

        assert not result.isError
        assert len(result.content) == 1

    async def test_get_hotspots_single_result(
        self, mcp_server: MCPVectorSearchServer
    ) -> None:
        """Test getting complexity hotspots limited to 1."""
        args = {"limit": 1}

        result = await mcp_server._get_complexity_hotspots(args)

        assert not result.isError
        assert len(result.content) == 1


@pytest.mark.asyncio
@pytest.mark.skip(
    reason="Obsolete tests after v1.2.7 refactoring - MCP server handlers extracted to separate modules"
)
class TestCheckCircularDependencies:
    """Tests for check_circular_dependencies tool."""

    async def test_check_circular_dependencies_none(
        self, mcp_server: MCPVectorSearchServer
    ) -> None:
        """Test circular dependency check with no cycles."""
        args: dict[str, Any] = {}

        result = await mcp_server._check_circular_dependencies(args)

        assert not result.isError
        assert len(result.content) == 1
        response_text = result.content[0].text
        # Simple project should have no circular dependencies
        assert "No circular dependencies detected" in response_text

    async def test_check_circular_dependencies_with_cycles(
        self, tmp_path: Path
    ) -> None:
        """Test circular dependency check with artificial cycles."""
        # Create files with circular imports
        project_dir = tmp_path / "circular_project"
        project_dir.mkdir()

        file_a = project_dir / "module_a.py"
        file_a.write_text("from module_b import func_b\n\ndef func_a(): pass\n")

        file_b = project_dir / "module_b.py"
        file_b.write_text("from module_a import func_a\n\ndef func_b(): pass\n")

        server = MCPVectorSearchServer(project_root=project_dir)
        args: dict[str, Any] = {}

        result = await server._check_circular_dependencies(args)

        assert not result.isError
        assert len(result.content) == 1
        # May detect cycles depending on import graph implementation
        # Just verify it doesn't crash


@pytest.mark.asyncio
@pytest.mark.skip(
    reason="Obsolete tests after v1.2.7 refactoring - MCP server handlers extracted to separate modules"
)
class TestThresholdPresets:
    """Tests for threshold configuration presets."""

    async def test_strict_preset(self, mcp_server: MCPVectorSearchServer) -> None:
        """Test strict threshold preset configuration."""
        config = mcp_server._get_threshold_config("strict")

        assert config.complexity.cognitive_a == 3
        assert config.smells.long_method_lines == 30
        assert config.smells.too_many_parameters == 3

    async def test_standard_preset(self, mcp_server: MCPVectorSearchServer) -> None:
        """Test standard threshold preset configuration."""
        config = mcp_server._get_threshold_config("standard")

        # Should use default ThresholdConfig values
        assert config.complexity.cognitive_a == 5
        assert config.smells.long_method_lines == 50
        assert config.smells.too_many_parameters == 5

    async def test_relaxed_preset(self, mcp_server: MCPVectorSearchServer) -> None:
        """Test relaxed threshold preset configuration."""
        config = mcp_server._get_threshold_config("relaxed")

        assert config.complexity.cognitive_a == 7
        assert config.smells.long_method_lines == 75
        assert config.smells.too_many_parameters == 7


@pytest.mark.asyncio
@pytest.mark.skip(
    reason="Obsolete tests after v1.2.7 refactoring - MCP server handlers extracted to separate modules"
)
class TestErrorHandling:
    """Tests for error handling in analysis tools."""

    async def test_analyze_project_with_invalid_files(self, tmp_path: Path) -> None:
        """Test analyze_project handles invalid files gracefully."""
        project_dir = tmp_path / "invalid_project"
        project_dir.mkdir()

        # Create invalid Python file
        invalid_file = project_dir / "invalid.py"
        invalid_file.write_text("this is not valid python @@@ ###")

        server = MCPVectorSearchServer(project_root=project_dir)
        args = {"threshold_preset": "standard", "output_format": "summary"}

        # Should handle parsing errors gracefully
        result = await server._analyze_project(args)

        # May succeed with no files analyzed, or return error
        assert len(result.content) == 1

    async def test_analyze_file_with_parsing_error(self, tmp_path: Path) -> None:
        """Test analyze_file handles parsing errors."""
        project_dir = tmp_path / "parse_error_project"
        project_dir.mkdir()

        invalid_file = project_dir / "invalid.py"
        invalid_file.write_text("def broken_syntax(")

        server = MCPVectorSearchServer(project_root=project_dir)
        args = {"file_path": "invalid.py"}

        result = await server._analyze_file(args)

        # Should handle parsing error gracefully
        assert len(result.content) == 1
