"""Search history and favorites management."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger
from rich.console import Console
from rich.table import Table

from .output import print_error, print_info, print_success

console = Console()


class SearchHistory:
    """Manage search history and favorites."""

    def __init__(self, project_root: Path):
        """Initialize search history manager."""
        self.project_root = project_root
        self.history_file = project_root / ".mcp-vector-search" / "search_history.json"
        self.favorites_file = project_root / ".mcp-vector-search" / "favorites.json"

        # Ensure directory exists
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

    def add_search(
        self,
        query: str,
        results_count: int,
        filters: dict[str, Any] | None = None,
        execution_time: float | None = None,
    ) -> None:
        """Add a search to history.

        Args:
            query: Search query
            results_count: Number of results found
            filters: Applied filters
            execution_time: Search execution time in seconds
        """
        try:
            history = self._load_history()

            search_entry = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "results_count": results_count,
                "filters": filters or {},
                "execution_time": execution_time,
            }

            # Add to beginning of history
            history.insert(0, search_entry)

            # Keep only last 100 searches
            history = history[:100]

            self._save_history(history)

        except Exception as e:
            print_error(f"Failed to save search history: {e}")

    def get_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get search history.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of search history entries
        """
        try:
            history = self._load_history()
            return history[:limit]
        except Exception as e:
            print_error(f"Failed to load search history: {e}")
            return []

    def clear_history(self) -> bool:
        """Clear search history.

        Returns:
            True if successful
        """
        try:
            self._save_history([])
            print_success("Search history cleared")
            return True
        except Exception as e:
            print_error(f"Failed to clear search history: {e}")
            return False

    def add_favorite(self, query: str, description: str | None = None) -> bool:
        """Add a search query to favorites.

        Args:
            query: Search query to favorite
            description: Optional description

        Returns:
            True if successful
        """
        try:
            favorites = self._load_favorites()

            # Check if already exists
            for fav in favorites:
                if fav["query"] == query:
                    print_info(f"Query already in favorites: {query}")
                    return True

            favorite_entry = {
                "query": query,
                "description": description or "",
                "created": datetime.now().isoformat(),
                "usage_count": 0,
            }

            favorites.append(favorite_entry)
            self._save_favorites(favorites)

            print_success(f"Added to favorites: {query}")
            return True

        except Exception as e:
            print_error(f"Failed to add favorite: {e}")
            return False

    def remove_favorite(self, query: str) -> bool:
        """Remove a query from favorites.

        Args:
            query: Query to remove

        Returns:
            True if successful
        """
        try:
            favorites = self._load_favorites()
            original_count = len(favorites)

            favorites = [fav for fav in favorites if fav["query"] != query]

            if len(favorites) < original_count:
                self._save_favorites(favorites)
                print_success(f"Removed from favorites: {query}")
                return True
            else:
                print_info(f"Query not found in favorites: {query}")
                return False

        except Exception as e:
            print_error(f"Failed to remove favorite: {e}")
            return False

    def get_favorites(self) -> list[dict[str, Any]]:
        """Get favorite queries.

        Returns:
            List of favorite queries
        """
        try:
            return self._load_favorites()
        except Exception as e:
            print_error(f"Failed to load favorites: {e}")
            return []

    def increment_favorite_usage(self, query: str) -> None:
        """Increment usage count for a favorite query.

        Args:
            query: Query that was used
        """
        try:
            favorites = self._load_favorites()

            for fav in favorites:
                if fav["query"] == query:
                    fav["usage_count"] = fav.get("usage_count", 0) + 1
                    fav["last_used"] = datetime.now().isoformat()
                    break

            self._save_favorites(favorites)

        except Exception as e:
            # Don't show error for this non-critical operation
            logger.debug(f"Failed to update history ranking: {e}")
            pass

    def _load_history(self) -> list[dict[str, Any]]:
        """Load search history from file."""
        if not self.history_file.exists():
            return []

        try:
            with open(self.history_file, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.debug(f"Failed to load history file: {e}")
            return []

    def _save_history(self, history: list[dict[str, Any]]) -> None:
        """Save search history to file."""
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def _load_favorites(self) -> list[dict[str, Any]]:
        """Load favorites from file."""
        if not self.favorites_file.exists():
            return []

        try:
            with open(self.favorites_file, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.debug(f"Failed to load favorites file: {e}")
            return []

    def _save_favorites(self, favorites: list[dict[str, Any]]) -> None:
        """Save favorites to file."""
        with open(self.favorites_file, "w", encoding="utf-8") as f:
            json.dump(favorites, f, indent=2, ensure_ascii=False)


def show_search_history(project_root: Path, limit: int = 20) -> None:
    """Display search history in a formatted table."""
    history_manager = SearchHistory(project_root)
    history = history_manager.get_history(limit)

    if not history:
        print_info("No search history found")
        return

    table = Table(
        title=f"Search History (Last {len(history)} searches)", show_header=True
    )
    table.add_column("#", style="cyan", width=3)
    table.add_column("Query", style="white", min_width=20)
    table.add_column("Results", style="green", width=8)
    table.add_column("Time", style="dim", width=16)
    table.add_column("Filters", style="yellow", width=15)

    for i, entry in enumerate(history, 1):
        timestamp = datetime.fromisoformat(entry["timestamp"]).strftime("%m-%d %H:%M")
        filters_str = ", ".join(f"{k}:{v}" for k, v in entry.get("filters", {}).items())
        if not filters_str:
            filters_str = "-"

        table.add_row(
            str(i),
            entry["query"][:40] + "..." if len(entry["query"]) > 40 else entry["query"],
            str(entry["results_count"]),
            timestamp,
            filters_str[:15] + "..." if len(filters_str) > 15 else filters_str,
        )

    console.print(table)


def show_favorites(project_root: Path) -> None:
    """Display favorite queries in a formatted table."""
    history_manager = SearchHistory(project_root)
    favorites = history_manager.get_favorites()

    if not favorites:
        print_info("No favorite queries found")
        return

    # Sort by usage count (descending)
    favorites.sort(key=lambda x: x.get("usage_count", 0), reverse=True)

    table = Table(title="Favorite Queries", show_header=True)
    table.add_column("#", style="cyan", width=3)
    table.add_column("Query", style="white", min_width=25)
    table.add_column("Description", style="dim", min_width=20)
    table.add_column("Usage", style="green", width=6)
    table.add_column("Created", style="dim", width=10)

    for i, fav in enumerate(favorites, 1):
        created = datetime.fromisoformat(fav["created"]).strftime("%m-%d")
        description = fav.get("description", "")[:30]
        if len(fav.get("description", "")) > 30:
            description += "..."

        table.add_row(
            str(i),
            fav["query"],
            description or "-",
            str(fav.get("usage_count", 0)),
            created,
        )

    console.print(table)
