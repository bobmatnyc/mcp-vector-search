"""Integration tests for MCP functionality."""

import tempfile
from pathlib import Path

import pytest

from mcp_vector_search.core.project import ProjectManager
from mcp_vector_search.mcp.server import MCPVectorSearchServer, create_mcp_server


class TestMCPIntegration:
    """Test MCP server integration."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)

            # Create some test files
            (project_root / "test.py").write_text(
                """
def hello_world():
    '''A simple hello world function.'''
    print('Hello, World!')

def add_numbers(a, b):
    '''Add two numbers together.'''
    return a + b

class Calculator:
    '''A simple calculator class.'''

    def multiply(self, x, y):
        '''Multiply two numbers.'''
        return x * y

    def divide(self, x, y):
        '''Divide two numbers.'''
        if y == 0:
            raise ValueError("Cannot divide by zero")
        return x / y
"""
            )

            (project_root / "utils.js").write_text(
                """
function formatString(str) {
    // Format a string with proper capitalization
    return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
}

function validateEmail(email) {
    // Simple email validation
    const emailRegex = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;
    return emailRegex.test(email);
}
"""
            )

            yield project_root

    async def _initialize_project(self, project_root):
        """Initialize a project with indexing."""
        # Initialize project
        project_manager = ProjectManager(project_root)
        project_manager.initialize(
            file_extensions=[".py", ".js"],
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )

        # Index the project
        from mcp_vector_search.cli.commands.index import run_indexing

        await run_indexing(
            project_root=project_root, force_reindex=True, show_progress=False
        )

        return project_root

    @pytest.mark.asyncio
    async def test_mcp_server_initialization(self, temp_project):
        """Test MCP server can be initialized."""
        initialized_project = await self._initialize_project(temp_project)
        server = MCPVectorSearchServer(initialized_project)

        await server.initialize()
        assert server._initialized
        assert server.search_engine is not None

        await server.cleanup()

    @pytest.mark.asyncio
    async def test_mcp_server_tools(self, temp_project):
        """Test MCP server provides correct tools."""
        initialized_project = await self._initialize_project(temp_project)
        server = MCPVectorSearchServer(initialized_project)
        tools = server.get_tools()

        tool_names = [tool.name for tool in tools]
        assert "search_code" in tool_names
        assert "get_project_status" in tool_names
        assert "index_project" in tool_names

    @pytest.mark.asyncio
    async def test_search_code_tool(self, temp_project):
        """Test the search_code tool."""
        initialized_project = await self._initialize_project(temp_project)
        server = MCPVectorSearchServer(initialized_project)
        await server.initialize()

        # Create a mock request
        class MockRequest:
            def __init__(self, name, arguments):
                self.params = MockParams(name, arguments)

        class MockParams:
            def __init__(self, name, arguments):
                self.name = name
                self.arguments = arguments

        # Test search
        request = MockRequest("search_code", {"query": "hello world", "limit": 5})
        result = await server.call_tool(request)

        assert not result.isError
        assert len(result.content) > 0
        assert "hello world" in result.content[0].text.lower()

        await server.cleanup()

    @pytest.mark.asyncio
    async def test_get_project_status_tool(self, temp_project):
        """Test the get_project_status tool."""
        initialized_project = await self._initialize_project(temp_project)
        server = MCPVectorSearchServer(initialized_project)
        await server.initialize()

        # Create a mock request
        class MockRequest:
            def __init__(self, name, arguments):
                self.params = MockParams(name, arguments)

        class MockParams:
            def __init__(self, name, arguments):
                self.name = name
                self.arguments = arguments

        # Test status
        request = MockRequest("get_project_status", {})
        result = await server.call_tool(request)

        assert not result.isError
        assert len(result.content) > 0
        status_text = result.content[0].text
        assert "Project Status" in status_text
        assert str(initialized_project) in status_text

        await server.cleanup()

    def test_mcp_server_creation(self):
        """Test MCP server can be created."""
        server = create_mcp_server()
        assert server is not None
        assert hasattr(server, "_mcp_server")

    def test_claude_code_commands_available(self):
        """Test that Claude Code commands are available."""
        from mcp_vector_search.cli.commands.mcp import (
            check_claude_code_available,
            get_claude_command,
        )

        # This will depend on the test environment
        # In CI, Claude Code might not be available
        claude_available = check_claude_code_available()

        if claude_available:
            claude_cmd = get_claude_command()
            assert claude_cmd is not None
            assert "claude" in claude_cmd

    def test_mcp_server_command_generation(self):
        """Test MCP server command generation."""
        from mcp_vector_search.cli.commands.mcp import get_mcp_server_command

        project_root = Path("/test/project")
        command = get_mcp_server_command(project_root)

        assert "python" in command
        assert "mcp_vector_search.mcp.server" in command
        assert str(project_root) in command


@pytest.mark.asyncio
async def test_mcp_server_stdio_protocol():
    """Test MCP server can handle stdio protocol messages."""
    # This is a more complex test that would require setting up
    # the full MCP protocol communication
    pass


if __name__ == "__main__":
    pytest.main([__file__])
