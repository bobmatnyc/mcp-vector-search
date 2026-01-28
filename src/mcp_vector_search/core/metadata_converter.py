"""Metadata conversion between CodeChunk and ChromaDB formats."""

import json
from pathlib import Path
from typing import Any

from .models import CodeChunk


class MetadataConverter:
    """Converts between CodeChunk objects and ChromaDB metadata format.

    Handles serialization of complex fields (lists, dicts) to JSON strings
    for ChromaDB storage and deserialization back to Python objects.
    """

    @staticmethod
    def chunk_to_metadata(
        chunk: CodeChunk, structural_metrics: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Convert CodeChunk to ChromaDB metadata format.

        Args:
            chunk: Code chunk to convert
            structural_metrics: Optional structural metrics to merge

        Returns:
            Metadata dictionary compatible with ChromaDB
        """
        metadata = {
            "file_path": str(chunk.file_path),
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "language": chunk.language,
            "chunk_type": chunk.chunk_type,
            "function_name": chunk.function_name or "",
            "class_name": chunk.class_name or "",
            "docstring": chunk.docstring or "",
            "complexity_score": chunk.complexity_score,
            # Hierarchy fields (convert lists to JSON strings for ChromaDB)
            "chunk_id": chunk.chunk_id or "",
            "parent_chunk_id": chunk.parent_chunk_id or "",
            "child_chunk_ids": json.dumps(chunk.child_chunk_ids or []),
            "chunk_depth": chunk.chunk_depth,
            # Additional metadata (convert lists/dicts to JSON strings)
            "decorators": json.dumps(chunk.decorators or []),
            "parameters": json.dumps(chunk.parameters or []),
            "return_type": chunk.return_type or "",
            "type_annotations": json.dumps(chunk.type_annotations or {}),
            # Monorepo support
            "subproject_name": chunk.subproject_name or "",
            "subproject_path": chunk.subproject_path or "",
        }

        # Merge structural metrics if provided
        if (
            structural_metrics
            and chunk.chunk_id
            and chunk.chunk_id in structural_metrics
        ):
            chunk_metrics = structural_metrics[chunk.chunk_id]
            metadata.update(chunk_metrics)

        return metadata

    @staticmethod
    def metadata_to_chunk(metadata: dict[str, Any], content: str) -> CodeChunk:
        """Convert ChromaDB metadata back to CodeChunk object.

        Args:
            metadata: ChromaDB metadata dictionary
            content: Code content

        Returns:
            Reconstructed CodeChunk object
        """
        # Parse JSON strings back to lists/dicts
        child_chunk_ids = metadata.get("child_chunk_ids", "[]")
        if isinstance(child_chunk_ids, str):
            child_chunk_ids = json.loads(child_chunk_ids)

        decorators = metadata.get("decorators", "[]")
        if isinstance(decorators, str):
            decorators = json.loads(decorators)

        parameters = metadata.get("parameters", "[]")
        if isinstance(parameters, str):
            parameters = json.loads(parameters)

        type_annotations = metadata.get("type_annotations", "{}")
        if isinstance(type_annotations, str):
            type_annotations = json.loads(type_annotations)

        return CodeChunk(
            content=content,
            file_path=Path(metadata["file_path"]),
            start_line=metadata["start_line"],
            end_line=metadata["end_line"],
            language=metadata["language"],
            chunk_type=metadata.get("chunk_type", "code"),
            function_name=metadata.get("function_name") or None,
            class_name=metadata.get("class_name") or None,
            docstring=metadata.get("docstring") or None,
            imports=metadata.get("imports", []),
            complexity_score=metadata.get("complexity_score", 0.0),
            chunk_id=metadata.get("chunk_id") or None,
            parent_chunk_id=metadata.get("parent_chunk_id") or None,
            child_chunk_ids=child_chunk_ids,
            chunk_depth=metadata.get("chunk_depth", 0),
            decorators=decorators,
            parameters=parameters,
            return_type=metadata.get("return_type") or None,
            type_annotations=type_annotations,
            subproject_name=metadata.get("subproject_name") or None,
            subproject_path=metadata.get("subproject_path") or None,
        )

    @staticmethod
    def create_searchable_text(chunk: CodeChunk) -> str:
        """Create optimized searchable text from code chunk.

        Combines code content with contextual information for better
        semantic search results.

        Args:
            chunk: Code chunk to create searchable text from

        Returns:
            Enhanced searchable text
        """
        parts = [chunk.content]

        # Add contextual information
        if chunk.function_name:
            parts.append(f"Function: {chunk.function_name}")

        if chunk.class_name:
            parts.append(f"Class: {chunk.class_name}")

        if chunk.docstring:
            parts.append(f"Documentation: {chunk.docstring}")

        # Add language and file context
        parts.append(f"Language: {chunk.language}")
        parts.append(f"File: {chunk.file_path.name}")

        return "\n".join(parts)
