"""Integration tests for NLP entity extraction in parsing pipeline."""

import pytest

from mcp_vector_search.parsers.python import PythonParser


class TestNLPIntegration:
    """Test NLP entity extraction integrated with parsers."""

    @pytest.fixture
    def python_parser(self):
        """Create Python parser instance."""
        return PythonParser()

    @pytest.mark.asyncio
    async def test_python_parser_extracts_nlp_entities(self, python_parser, tmp_path):
        """Test that Python parser extracts NLP entities from docstrings."""
        # Create a test Python file
        test_file = tmp_path / "test_code.py"
        test_file.write_text(
            '''
"""Module for vector search operations."""

class VectorDatabase:
    """Manages vector embeddings using LanceDB.

    Stores code chunks with metadata and performs similarity search
    using `sentence-transformers` embeddings.

    Returns:
        SearchResult objects ranked by score
    """

    def search(self, query: str, limit: int = 10):
        """Perform semantic search on code.

        Generates embeddings and queries the vector database.

        Args:
            query: Search query string
            limit: Maximum results

        Returns:
            List of SearchResult objects

        Raises:
            SearchError: If database unavailable
        """
        pass
'''
        )

        # Parse the file
        chunks = await python_parser.parse_file(test_file)

        # Find the class chunk
        class_chunk = next((c for c in chunks if c.chunk_type == "class"), None)
        assert class_chunk is not None
        assert class_chunk.class_name == "VectorDatabase"

        # Verify NLP entities were extracted from class docstring
        assert hasattr(class_chunk, "nlp_keywords")
        assert hasattr(class_chunk, "nlp_code_refs")
        assert hasattr(class_chunk, "nlp_technical_terms")

        # Check that we got some keywords (may vary based on YAKE)
        assert len(class_chunk.nlp_keywords) > 0

        # Check code references
        assert "sentence-transformers" in class_chunk.nlp_code_refs

        # Check technical terms (should find CamelCase/ACRONYMS)
        assert (
            "LanceDB" in class_chunk.nlp_technical_terms
            or "SearchResult" in class_chunk.nlp_technical_terms
        )

        # Find the function chunk
        function_chunk = next((c for c in chunks if c.function_name == "search"), None)
        assert function_chunk is not None

        # Verify function also has NLP entities
        assert len(function_chunk.nlp_keywords) > 0

        # Check action verbs were extracted
        # Note: action_verbs are not stored in chunks, only keywords/refs/terms
        # But we can verify keywords include docstring-relevant terms

    @pytest.mark.asyncio
    async def test_chunks_without_docstrings_have_empty_nlp_fields(
        self, python_parser, tmp_path
    ):
        """Test that chunks without docstrings have empty NLP fields."""
        # Create a test Python file without docstrings
        test_file = tmp_path / "no_docs.py"
        test_file.write_text(
            """
def simple_function(x):
    return x * 2

class SimpleClass:
    def method(self):
        pass
"""
        )

        # Parse the file
        chunks = await python_parser.parse_file(test_file)

        # All chunks should have empty NLP fields
        for chunk in chunks:
            assert hasattr(chunk, "nlp_keywords")
            assert hasattr(chunk, "nlp_code_refs")
            assert hasattr(chunk, "nlp_technical_terms")
            # Empty lists since no docstrings
            assert chunk.nlp_keywords == []
            assert chunk.nlp_code_refs == []
            assert chunk.nlp_technical_terms == []

    @pytest.mark.asyncio
    async def test_nlp_fields_serialization(self, python_parser, tmp_path):
        """Test that NLP fields are properly serialized in to_dict()."""
        test_file = tmp_path / "test_serialization.py"
        test_file.write_text(
            '''
def example():
    """Returns `DatabaseConnection` instance."""
    pass
'''
        )

        chunks = await python_parser.parse_file(test_file)
        function_chunk = next((c for c in chunks if c.function_name == "example"), None)
        assert function_chunk is not None

        # Convert to dict
        chunk_dict = function_chunk.to_dict()

        # Verify NLP fields are in dict
        assert "nlp_keywords" in chunk_dict
        assert "nlp_code_refs" in chunk_dict
        assert "nlp_technical_terms" in chunk_dict

        # Verify values
        assert isinstance(chunk_dict["nlp_keywords"], list)
        assert isinstance(chunk_dict["nlp_code_refs"], list)
        assert isinstance(chunk_dict["nlp_technical_terms"], list)

        # Should have extracted code reference
        assert "DatabaseConnection" in chunk_dict["nlp_code_refs"]
