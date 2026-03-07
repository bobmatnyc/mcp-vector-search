"""Unit tests for KG scoped entity resolver — self.method() / cls.method() resolution."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mcp_vector_search.core.kg_builder import KGBuilder
from mcp_vector_search.core.models import CodeChunk


def make_method_chunk(
    function_name: str,
    class_name: str,
    chunk_id: str,
    file_path: str = "/repo/src/engine.py",
) -> CodeChunk:
    """Create a minimal CodeChunk representing a class method."""
    return CodeChunk(
        content=f"def {function_name}(self): pass",
        file_path=Path(file_path),
        start_line=1,
        end_line=1,
        language="python",
        chunk_type="function",
        function_name=function_name,
        class_name=class_name,
        chunk_id=chunk_id,
        calls=[],
    )


@pytest.fixture
def builder() -> KGBuilder:
    """Create a KGBuilder with a mock KG."""
    mock_kg = MagicMock()
    return KGBuilder(kg=mock_kg, project_root=Path("/repo"))


@pytest.fixture
def preloaded_builder(builder: KGBuilder) -> KGBuilder:
    """Builder pre-scanned with SemanticSearchEngine methods."""
    chunks = [
        make_method_chunk("search", "SemanticSearchEngine", "chunk_search"),
        make_method_chunk("_build", "SemanticSearchEngine", "chunk_build"),
        make_method_chunk("index", "OtherClass", "chunk_index"),
    ]
    builder._prescan_entity_names(chunks)
    return builder


class TestClassMemberMapPopulation:
    def test_class_member_map_populated_for_method_chunk(self, builder: KGBuilder):
        """_class_member_map is populated during prescan for chunks with class_name."""
        chunks = [make_method_chunk("search", "SemanticSearchEngine", "chunk_search")]
        builder._prescan_entity_names(chunks)

        assert "SemanticSearchEngine" in builder._class_member_map
        assert (
            builder._class_member_map["SemanticSearchEngine"]["search"]
            == "chunk_search"
        )

    def test_class_member_map_cleared_on_repopulation(self, builder: KGBuilder):
        """_class_member_map is cleared when _prescan_entity_names is called again."""
        builder._class_member_map["StaleClass"] = {"old_method": "stale_id"}

        chunks = [make_method_chunk("search", "SemanticSearchEngine", "chunk_search")]
        builder._prescan_entity_names(chunks)

        assert "StaleClass" not in builder._class_member_map

    def test_class_member_map_multiple_methods(self, builder: KGBuilder):
        """Multiple methods on the same class are all tracked."""
        chunks = [
            make_method_chunk("search", "SemanticSearchEngine", "chunk_search"),
            make_method_chunk("_build", "SemanticSearchEngine", "chunk_build"),
        ]
        builder._prescan_entity_names(chunks)

        assert (
            builder._class_member_map["SemanticSearchEngine"]["search"]
            == "chunk_search"
        )
        assert (
            builder._class_member_map["SemanticSearchEngine"]["_build"] == "chunk_build"
        )

    def test_class_member_map_not_populated_for_bare_function(self, builder: KGBuilder):
        """Chunks without class_name are not added to _class_member_map."""
        chunk = CodeChunk(
            content="def standalone(): pass",
            file_path=Path("/repo/src/utils.py"),
            start_line=1,
            end_line=1,
            language="python",
            chunk_type="function",
            function_name="standalone",
            class_name=None,
            chunk_id="chunk_standalone",
            calls=[],
        )
        builder._prescan_entity_names([chunk])

        assert builder._class_member_map == {}


class TestResolveScopedSelf:
    def test_resolves_self_method(self, preloaded_builder: KGBuilder):
        """self.search resolves to the correct chunk_id for the caller's class."""
        result = preloaded_builder._resolve_scoped(
            "self.search", "SemanticSearchEngine"
        )
        assert result == "chunk_search"

    def test_resolves_cls_method(self, preloaded_builder: KGBuilder):
        """cls._build resolves to the correct chunk_id for the caller's class."""
        result = preloaded_builder._resolve_scoped("cls._build", "SemanticSearchEngine")
        assert result == "chunk_build"

    def test_returns_none_for_wrong_class(self, preloaded_builder: KGBuilder):
        """self.method for a class that does not own the method returns None."""
        result = preloaded_builder._resolve_scoped("self.search", "WrongClass")
        assert result is None

    def test_returns_none_for_non_self_cls_prefix(self, preloaded_builder: KGBuilder):
        """Dotted names whose base is not self/cls fall through (return None)."""
        result = preloaded_builder._resolve_scoped(
            "other.search", "SemanticSearchEngine"
        )
        assert result is None

    def test_returns_none_when_caller_class_is_none(self, preloaded_builder: KGBuilder):
        """Without caller class context, scoped resolution always returns None."""
        result = preloaded_builder._resolve_scoped("self.search", None)
        assert result is None

    def test_returns_none_when_name_has_no_dot(self, preloaded_builder: KGBuilder):
        """Plain names without a dot are not handled by scoped resolver."""
        result = preloaded_builder._resolve_scoped("search", "SemanticSearchEngine")
        assert result is None

    def test_returns_none_when_method_not_in_class(self, preloaded_builder: KGBuilder):
        """self.nonexistent returns None — no edge created for unknown methods."""
        result = preloaded_builder._resolve_scoped(
            "self.nonexistent", "SemanticSearchEngine"
        )
        assert result is None


class TestScopedResolutionPrecedence:
    def test_scoped_wins_over_unique_resolver_for_self_calls(self, builder: KGBuilder):
        """self.method() correctly resolves to the class-scoped chunk even when
        the method name is ambiguous globally (shared by multiple classes)."""
        # Two classes each have a method named "execute" — ambiguous globally
        # but the scoped resolver knows which class the caller belongs to.
        chunks = [
            make_method_chunk("execute", "PipelineA", "chunk_execute_a"),
            make_method_chunk("execute", "PipelineB", "chunk_execute_b"),
        ]
        builder._prescan_entity_names(chunks)

        # Global resolver: "execute" is ambiguous → None
        assert builder._resolve_entity("execute") is None

        # Scoped resolver: caller is PipelineA → resolves correctly
        assert builder._resolve_scoped("self.execute", "PipelineA") == "chunk_execute_a"
        assert builder._resolve_scoped("self.execute", "PipelineB") == "chunk_execute_b"
