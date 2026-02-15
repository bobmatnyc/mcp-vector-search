"""Unit tests for NLP entity extractor."""

import pytest

from mcp_vector_search.core.nlp_extractor import ExtractedEntities, NLPExtractor


class TestNLPExtractor:
    """Test NLP entity extraction from docstrings."""

    @pytest.fixture
    def extractor(self):
        """Create NLP extractor instance."""
        return NLPExtractor(max_keywords=10)

    def test_extract_from_empty_text(self, extractor):
        """Test extraction from empty text returns empty entities."""
        entities = extractor.extract("")
        assert entities.keywords == []
        assert entities.code_references == []
        assert entities.technical_terms == []
        assert entities.action_verbs == []

    def test_extract_from_short_text(self, extractor):
        """Test extraction from very short text returns empty entities."""
        entities = extractor.extract("hi")
        assert entities.keywords == []
        assert entities.code_references == []
        assert entities.technical_terms == []
        assert entities.action_verbs == []

    def test_extract_keywords(self, extractor):
        """Test keyword extraction from docstring."""
        text = """
        Generate embeddings for code chunks using sentence transformers.
        This function processes text and creates vector representations.
        """
        entities = extractor.extract(text)
        assert len(entities.keywords) > 0
        # Keywords should be lowercased
        assert any("embedding" in kw.lower() for kw in entities.keywords)

    def test_extract_code_references(self, extractor):
        """Test extraction of backtick code references."""
        text = """
        Use `VectorDatabase` to store embeddings. Call `get_results()` to retrieve matches.
        The `CodeChunk` model contains metadata.
        """
        entities = extractor.extract(text)
        assert "VectorDatabase" in entities.code_references
        assert "get_results()" in entities.code_references
        assert "CodeChunk" in entities.code_references

    def test_extract_technical_terms(self, extractor):
        """Test extraction of technical terms (CamelCase, ACRONYMS, snake_case)."""
        text = """
        The DatabaseConnection class uses HTTP and HTTPS protocols.
        Configure vector_store and embedding_model parameters.
        """
        entities = extractor.extract(text)

        # Should find CamelCase
        assert "DatabaseConnection" in entities.technical_terms
        # Should find ACRONYMS
        assert "HTTP" in entities.technical_terms or "HTTPS" in entities.technical_terms
        # Should find snake_case
        assert (
            "vector_store" in entities.technical_terms
            or "embedding_model" in entities.technical_terms
        )

    def test_extract_action_verbs(self, extractor):
        """Test extraction of docstring action verbs."""
        text = """
        Returns the similarity score for the query.
        Raises ValueError if input is invalid.
        Creates a new embedding vector.
        Generates results asynchronously.
        """
        entities = extractor.extract(text)

        # Should find common docstring verbs
        assert "returns" in entities.action_verbs
        assert "raises" in entities.action_verbs
        assert "creates" in entities.action_verbs
        assert "generates" in entities.action_verbs

    def test_extract_realistic_docstring(self, extractor):
        """Test extraction from realistic Python docstring."""
        text = """
        Perform semantic search for code chunks.

        Uses `sentence-transformers` to generate embeddings and `LanceDB`
        for vector similarity search. Returns SearchResult objects ranked
        by similarity score.

        Args:
            query: Search query string
            limit: Maximum results to return
            filters: Optional metadata filters

        Returns:
            List of SearchResult objects

        Raises:
            SearchError: If database is unavailable
        """
        entities = extractor.extract(text)

        # Should have keywords
        assert len(entities.keywords) > 0

        # Should find code references
        assert "sentence-transformers" in entities.code_references
        assert "LanceDB" in entities.code_references

        # Should find technical terms
        assert "SearchResult" in entities.technical_terms

        # Should find action verbs
        assert "returns" in entities.action_verbs
        assert "raises" in entities.action_verbs

    def test_extract_handles_malformed_text(self, extractor):
        """Test extractor handles malformed/edge case text gracefully."""
        malformed_texts = [
            "```code block without closing backtick",
            "ALL CAPS TEXT WITHOUT MEANING",
            "special!@#$%characters^&*()",
            "\n\n\n\n\n",  # Only newlines
        ]

        for text in malformed_texts:
            entities = extractor.extract(text)
            # Should not crash, returns ExtractedEntities
            assert isinstance(entities, ExtractedEntities)
