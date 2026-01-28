"""Query builder for ChromaDB where clauses."""

from typing import Any


class QueryBuilder:
    """Builds ChromaDB where clauses from filter dictionaries.

    Supports various filter patterns:
    - Equality: {"language": "python"}
    - Negation: {"language": "!javascript"}
    - List (IN): {"language": ["python", "typescript"]}
    - Operators: {"cognitive_complexity": {"$gte": 10}}
    - Compound: {"$and": [...], "$or": [...]}
    """

    @staticmethod
    def build_where_clause(filters: dict[str, Any] | None) -> dict[str, Any] | None:
        """Build ChromaDB where clause from filters.

        Args:
            filters: Dictionary of filter criteria

        Returns:
            ChromaDB where clause or None if no filters

        Examples:
            >>> build_where_clause({"language": "python"})
            {"language": "python"}

            >>> build_where_clause({"language": ["python", "typescript"]})
            {"language": {"$in": ["python", "typescript"]}}

            >>> build_where_clause({"language": "!javascript"})
            {"language": {"$ne": "javascript"}}

            >>> build_where_clause({"cognitive_complexity": {"$gte": 10}})
            {"cognitive_complexity": {"$gte": 10}}
        """
        if not filters:
            return None

        # If filters already contain ChromaDB operators ($and, $or), pass through
        if "$and" in filters or "$or" in filters:
            return filters

        where = {}

        for key, value in filters.items():
            if isinstance(value, list):
                # List values -> $in operator
                where[key] = {"$in": value}
            elif isinstance(value, str) and value.startswith("!"):
                # Negation pattern -> $ne operator
                where[key] = {"$ne": value[1:]}
            elif isinstance(value, dict):
                # Already an operator query like {"$gte": 10}
                where[key] = value
            else:
                # Simple equality
                where[key] = value

        return where

    @staticmethod
    def build_simple_where_clause(
        filters: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Build simple ChromaDB where clause for basic filters.

        Simpler version that only handles common cases:
        - language, file_path, chunk_type equality
        - List values for $in queries

        Args:
            filters: Dictionary of filter criteria

        Returns:
            ChromaDB where clause or None if no filters
        """
        if not filters:
            return None

        conditions = []

        for key, value in filters.items():
            if key == "language" and value:
                conditions.append({"language": {"$eq": value}})
            elif key == "file_path" and value:
                if isinstance(value, list):
                    conditions.append({"file_path": {"$in": [str(p) for p in value]}})
                else:
                    conditions.append({"file_path": {"$eq": str(value)}})
            elif key == "chunk_type" and value:
                conditions.append({"chunk_type": {"$eq": value}})

        if not conditions:
            return None
        elif len(conditions) > 1:
            return {"$and": conditions}
        else:
            return conditions[0]
