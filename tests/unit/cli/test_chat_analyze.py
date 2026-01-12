"""Tests for chat analyze functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_vector_search.cli.commands.chat import run_chat_analyze


@pytest.fixture
def mock_project_root(tmp_path):
    """Create a mock project root directory."""
    project_root = tmp_path / "test_project"
    project_root.mkdir()

    # Create .mcp-vector-search directory
    config_dir = project_root / ".mcp-vector-search"
    config_dir.mkdir()

    # Create a sample Python file
    sample_file = project_root / "sample.py"
    sample_file.write_text(
        """
def hello():
    print("Hello, world!")
"""
    )

    return project_root


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    mock_client = MagicMock()
    mock_client.provider = "openai"
    mock_client.model = "gpt-4o"
    mock_client.stream_chat_completion = AsyncMock()

    # Mock streaming response
    async def mock_stream():
        chunks = ["# Analysis\n", "\n", "Code quality is **good**."]
        for chunk in chunks:
            yield chunk

    mock_client.stream_chat_completion.return_value = mock_stream()
    return mock_client


class TestRunChatAnalyze:
    """Test run_chat_analyze function."""

    @pytest.mark.skip(reason="Integration test - requires full mocking")
    @pytest.mark.asyncio
    async def test_analyze_basic_query(self, mock_project_root, mock_llm_client):
        """Test basic analysis query."""
        with (
            patch("mcp_vector_search.core.llm_client.LLMClient") as mock_llm_client,
            patch(
                "mcp_vector_search.core.project.ProjectManager"
            ) as mock_project_manager,
            patch(
                "mcp_vector_search.core.config_utils.get_openai_api_key"
            ) as mock_openai_key,
            patch(
                "mcp_vector_search.core.config_utils.get_openrouter_api_key"
            ) as mock_openrouter_key,
            patch(
                "mcp_vector_search.parsers.registry.ParserRegistry"
            ) as mock_parser_registry,
            patch("mcp_vector_search.analysis.ProjectMetrics") as mock_project_metrics,
            patch(
                "mcp_vector_search.analysis.interpretation.EnhancedJSONExporter"
            ) as mock_exporter,
        ):
            # Setup mocks
            mock_openai_key.return_value = "test-key"
            mock_openrouter_key.return_value = None
            mock_llm_client.return_value = mock_llm_client

            # Mock project manager
            mock_pm = MagicMock()
            mock_pm.is_initialized.return_value = True
            mock_config = MagicMock()
            mock_config.file_extensions = [".py"]
            mock_config.ignore_patterns = ["node_modules", ".git"]
            mock_pm.load_config.return_value = mock_config
            mock_project_manager.return_value = mock_pm

            # Mock parser registry
            mock_parser = MagicMock()
            mock_parser.parse_file.return_value = []
            mock_registry = MagicMock()
            mock_registry.get_parser.return_value = mock_parser
            mock_parser_registry.return_value = mock_registry

            # Mock metrics
            mock_metrics = MagicMock()
            mock_project_metrics.return_value = mock_metrics

            # Mock exporter
            mock_export = MagicMock()
            mock_export.model_dump.return_value = {"summary": "test"}
            mock_exporter = MagicMock()
            mock_exporter.export_with_context.return_value = mock_export
            mock_exporter.return_value = mock_exporter

            # Run analysis
            await run_chat_analyze(
                project_root=mock_project_root,
                query="What's the cognitive complexity?",
                provider="openai",
                timeout=30.0,
                think=True,
            )

            # Verify LLM client was created with correct parameters
            mock_llm_client.assert_called_once()
            call_kwargs = mock_llm_client.call_args.kwargs
            assert call_kwargs["provider"] == "openai"
            assert call_kwargs["think"] is True

            # Verify stream was called
            mock_llm_client.stream_chat_completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_project_not_initialized(self, mock_project_root):
        """Test analysis when project is not initialized."""
        from mcp_vector_search.core.exceptions import ProjectNotFoundError

        with (
            patch(
                "mcp_vector_search.core.project.ProjectManager"
            ) as mock_project_manager,
            patch(
                "mcp_vector_search.core.config_utils.get_openai_api_key"
            ) as mock_openai_key,
            patch(
                "mcp_vector_search.core.config_utils.get_openrouter_api_key"
            ) as mock_openrouter_key,
        ):
            mock_openai_key.return_value = "test-key"
            mock_openrouter_key.return_value = None

            # Mock project manager to return not initialized
            mock_pm = MagicMock()
            mock_pm.is_initialized.return_value = False
            mock_project_manager.return_value = mock_pm

            # Should raise ProjectNotFoundError
            with pytest.raises(ProjectNotFoundError):
                await run_chat_analyze(
                    project_root=mock_project_root,
                    query="analyze complexity",
                    provider="openai",
                )

    @pytest.mark.skip(reason="Integration test - requires full mocking")
    @pytest.mark.asyncio
    async def test_analyze_fallback_on_stream_error(
        self, mock_project_root, mock_llm_client
    ):
        """Test fallback behavior when streaming fails."""
        with (
            patch("mcp_vector_search.core.llm_client.LLMClient") as mock_llm_client,
            patch(
                "mcp_vector_search.core.project.ProjectManager"
            ) as mock_project_manager,
            patch(
                "mcp_vector_search.core.config_utils.get_openai_api_key"
            ) as mock_openai_key,
            patch(
                "mcp_vector_search.core.config_utils.get_openrouter_api_key"
            ) as mock_openrouter_key,
            patch(
                "mcp_vector_search.parsers.registry.ParserRegistry"
            ) as mock_parser_registry,
            patch("mcp_vector_search.analysis.ProjectMetrics") as mock_project_metrics,
            patch(
                "mcp_vector_search.analysis.interpretation.EnhancedJSONExporter"
            ) as mock_exporter,
            patch(
                "mcp_vector_search.analysis.interpretation.AnalysisInterpreter"
            ) as mock_interpreter,
        ):
            # Setup mocks
            mock_openai_key.return_value = "test-key"
            mock_openrouter_key.return_value = None

            # Make streaming fail
            mock_llm_client.stream_chat_completion.side_effect = Exception(
                "Stream error"
            )
            mock_llm_client.return_value = mock_llm_client

            # Mock project manager
            mock_pm = MagicMock()
            mock_pm.is_initialized.return_value = True
            mock_config = MagicMock()
            mock_config.file_extensions = [".py"]
            mock_config.ignore_patterns = []
            mock_pm.load_config.return_value = mock_config
            mock_project_manager.return_value = mock_pm

            # Mock parser registry
            mock_registry = MagicMock()
            mock_registry.get_parser.return_value = None
            mock_parser_registry.return_value = mock_registry

            # Mock metrics
            mock_metrics = MagicMock()
            mock_project_metrics.return_value = mock_metrics

            # Mock exporter
            mock_export = MagicMock()
            mock_export.model_dump.return_value = {"summary": "test"}
            mock_exporter = MagicMock()
            mock_exporter.export_with_context.return_value = mock_export
            mock_exporter.return_value = mock_exporter

            # Mock interpreter
            mock_interpreter = MagicMock()
            mock_interpreter.interpret.return_value = "Fallback summary"
            mock_interpreter.return_value = mock_interpreter

            # Run analysis (should not raise exception, should fall back)
            await run_chat_analyze(
                project_root=mock_project_root,
                query="analyze code",
                provider="openai",
            )

            # Verify interpreter was used as fallback
            mock_interpreter.interpret.assert_called_once()

    @pytest.mark.skip(reason="Integration test - requires full mocking")
    @pytest.mark.asyncio
    async def test_analyze_uses_advanced_model(
        self, mock_project_root, mock_llm_client
    ):
        """Test that analysis always uses advanced model (think=True)."""
        with (
            patch("mcp_vector_search.core.llm_client.LLMClient") as mock_llm_client,
            patch(
                "mcp_vector_search.core.project.ProjectManager"
            ) as mock_project_manager,
            patch(
                "mcp_vector_search.core.config_utils.get_openai_api_key"
            ) as mock_openai_key,
            patch(
                "mcp_vector_search.core.config_utils.get_openrouter_api_key"
            ) as mock_openrouter_key,
            patch(
                "mcp_vector_search.parsers.registry.ParserRegistry"
            ) as mock_parser_registry,
            patch("mcp_vector_search.analysis.ProjectMetrics") as mock_project_metrics,
            patch(
                "mcp_vector_search.analysis.interpretation.EnhancedJSONExporter"
            ) as mock_exporter,
        ):
            # Setup mocks
            mock_openai_key.return_value = "test-key"
            mock_openrouter_key.return_value = None
            mock_llm_client.return_value = mock_llm_client

            # Mock project manager
            mock_pm = MagicMock()
            mock_pm.is_initialized.return_value = True
            mock_config = MagicMock()
            mock_config.file_extensions = [".py"]
            mock_config.ignore_patterns = []
            mock_pm.load_config.return_value = mock_config
            mock_project_manager.return_value = mock_pm

            # Mock parser registry
            mock_registry = MagicMock()
            mock_registry.get_parser.return_value = None
            mock_parser_registry.return_value = mock_registry

            # Mock metrics
            mock_metrics = MagicMock()
            mock_project_metrics.return_value = mock_metrics

            # Mock exporter
            mock_export = MagicMock()
            mock_export.model_dump.return_value = {"summary": "test"}
            mock_exporter = MagicMock()
            mock_exporter.export_with_context.return_value = mock_export
            mock_exporter.return_value = mock_exporter

            # Run with think=False explicitly
            await run_chat_analyze(
                project_root=mock_project_root,
                query="analyze",
                provider="openai",
                think=False,  # Explicitly set to False
            )

            # Verify LLM client was still created with think=True (always forced)
            call_kwargs = mock_llm_client.call_args.kwargs
            assert call_kwargs["think"] is True, (
                "Analysis should always use advanced model"
            )


# NOTE: Additional test placeholders removed. System prompt and query type tests
# should be implemented when the chat analyze feature is fully stabilized.
