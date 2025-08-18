"""Export functionality for search results."""

import csv
import json
from datetime import datetime
from pathlib import Path

from rich.console import Console

from ..core.models import SearchResult
from .output import print_error, print_success

console = Console()


class SearchResultExporter:
    """Export search results to various formats."""

    def __init__(self):
        """Initialize exporter."""
        pass

    def export_to_json(
        self,
        results: list[SearchResult],
        output_path: Path,
        query: str,
        include_metadata: bool = True,
    ) -> bool:
        """Export results to JSON format.

        Args:
            results: Search results to export
            output_path: Output file path
            query: Original search query
            include_metadata: Whether to include metadata

        Returns:
            True if successful
        """
        try:
            export_data = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "total_results": len(results),
                "results": [],
            }

            for result in results:
                result_data = {
                    "rank": result.rank,
                    "file_path": str(result.file_path),
                    "similarity_score": result.similarity_score,
                    "start_line": result.start_line,
                    "end_line": result.end_line,
                    "language": result.language,
                    "chunk_type": result.chunk_type,
                }

                if result.function_name:
                    result_data["function_name"] = result.function_name
                if result.class_name:
                    result_data["class_name"] = result.class_name
                if result.content:
                    result_data["content"] = result.content

                if include_metadata:
                    result_data["location"] = result.location

                export_data["results"].append(result_data)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            print_success(f"Exported {len(results)} results to {output_path}")
            return True

        except Exception as e:
            print_error(f"Failed to export to JSON: {e}")
            return False

    def export_to_csv(
        self, results: list[SearchResult], output_path: Path, query: str
    ) -> bool:
        """Export results to CSV format.

        Args:
            results: Search results to export
            output_path: Output file path
            query: Original search query

        Returns:
            True if successful
        """
        try:
            fieldnames = [
                "rank",
                "file_path",
                "similarity_score",
                "start_line",
                "end_line",
                "language",
                "chunk_type",
                "function_name",
                "class_name",
                "location",
            ]

            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                # Write metadata row
                writer.writerow(
                    {
                        "rank": f"# Query: {query}",
                        "file_path": f"# Timestamp: {datetime.now().isoformat()}",
                        "similarity_score": f"# Total Results: {len(results)}",
                        "start_line": "",
                        "end_line": "",
                        "language": "",
                        "chunk_type": "",
                        "function_name": "",
                        "class_name": "",
                        "location": "",
                    }
                )

                for result in results:
                    writer.writerow(
                        {
                            "rank": result.rank,
                            "file_path": str(result.file_path),
                            "similarity_score": f"{result.similarity_score:.4f}",
                            "start_line": result.start_line,
                            "end_line": result.end_line,
                            "language": result.language,
                            "chunk_type": result.chunk_type,
                            "function_name": result.function_name or "",
                            "class_name": result.class_name or "",
                            "location": result.location,
                        }
                    )

            print_success(f"Exported {len(results)} results to {output_path}")
            return True

        except Exception as e:
            print_error(f"Failed to export to CSV: {e}")
            return False

    def export_to_markdown(
        self,
        results: list[SearchResult],
        output_path: Path,
        query: str,
        include_content: bool = True,
    ) -> bool:
        """Export results to Markdown format.

        Args:
            results: Search results to export
            output_path: Output file path
            query: Original search query
            include_content: Whether to include code content

        Returns:
            True if successful
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                # Write header
                f.write("# Search Results\n\n")
                f.write(f"**Query:** `{query}`\n")
                f.write(
                    f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write(f"**Total Results:** {len(results)}\n\n")

                # Write results
                for result in results:
                    f.write(f"## {result.rank}. {result.file_path.name}\n\n")

                    # Basic info
                    f.write(f"- **File:** `{result.file_path}`\n")
                    f.write(
                        f"- **Location:** Lines {result.start_line}-{result.end_line}\n"
                    )
                    f.write(f"- **Similarity:** {result.similarity_score:.2%}\n")
                    f.write(f"- **Language:** {result.language}\n")

                    if result.function_name:
                        f.write(f"- **Function:** `{result.function_name}()`\n")
                    if result.class_name:
                        f.write(f"- **Class:** `{result.class_name}`\n")

                    f.write("\n")

                    # Code content
                    if include_content and result.content:
                        f.write(f"```{result.language}\n")
                        f.write(result.content)
                        f.write("\n```\n\n")

                    f.write("---\n\n")

            print_success(f"Exported {len(results)} results to {output_path}")
            return True

        except Exception as e:
            print_error(f"Failed to export to Markdown: {e}")
            return False

    def export_summary_table(
        self, results: list[SearchResult], output_path: Path, query: str
    ) -> bool:
        """Export a summary table of results.

        Args:
            results: Search results to export
            output_path: Output file path
            query: Original search query

        Returns:
            True if successful
        """
        try:
            # Calculate summary statistics
            languages = {}
            files = {}
            functions = {}
            classes = {}

            for result in results:
                languages[result.language] = languages.get(result.language, 0) + 1
                files[result.file_path.name] = files.get(result.file_path.name, 0) + 1

                if result.function_name:
                    functions[result.function_name] = (
                        functions.get(result.function_name, 0) + 1
                    )
                if result.class_name:
                    classes[result.class_name] = classes.get(result.class_name, 0) + 1

            avg_similarity = (
                sum(r.similarity_score for r in results) / len(results)
                if results
                else 0
            )

            summary_data = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_results": len(results),
                    "average_similarity": round(avg_similarity, 4),
                    "unique_files": len(files),
                    "unique_functions": len(functions),
                    "unique_classes": len(classes),
                },
                "distributions": {
                    "languages": dict(
                        sorted(languages.items(), key=lambda x: x[1], reverse=True)
                    ),
                    "top_files": dict(
                        sorted(files.items(), key=lambda x: x[1], reverse=True)[:10]
                    ),
                    "top_functions": dict(
                        sorted(functions.items(), key=lambda x: x[1], reverse=True)[:10]
                    ),
                    "top_classes": dict(
                        sorted(classes.items(), key=lambda x: x[1], reverse=True)[:10]
                    ),
                },
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)

            print_success(
                f"Exported summary for {len(results)} results to {output_path}"
            )
            return True

        except Exception as e:
            print_error(f"Failed to export summary: {e}")
            return False


def get_export_path(format_type: str, query: str, base_dir: Path | None = None) -> Path:
    """Generate export file path based on format and query.

    Args:
        format_type: Export format (json, csv, markdown, summary)
        query: Search query
        base_dir: Base directory for export

    Returns:
        Generated file path
    """
    if base_dir is None:
        base_dir = Path.cwd()

    # Sanitize query for filename
    safe_query = "".join(
        c for c in query if c.isalnum() or c in (" ", "-", "_")
    ).rstrip()
    safe_query = safe_query.replace(" ", "_")[:50]  # Limit length

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    extensions = {
        "json": ".json",
        "csv": ".csv",
        "markdown": ".md",
        "summary": "_summary.json",
    }

    filename = f"search_{safe_query}_{timestamp}{extensions.get(format_type, '.txt')}"
    return base_dir / filename
