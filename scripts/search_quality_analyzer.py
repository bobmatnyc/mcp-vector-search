#!/usr/bin/env python3
"""Search quality and relevance analyzer."""

import asyncio

# Add src to path for imports
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_vector_search.core.database import ChromaVectorDatabase
from mcp_vector_search.core.embeddings import create_embedding_function
from mcp_vector_search.core.indexer import SemanticIndexer
from mcp_vector_search.core.models import SearchResult
from mcp_vector_search.core.project import ProjectManager
from mcp_vector_search.core.search import SemanticSearchEngine


@dataclass
class QualityTestCase:
    """A test case for search quality assessment."""

    query: str
    expected_keywords: list[str]
    expected_file_patterns: list[str]
    expected_languages: list[str]
    min_results: int
    max_results: int
    min_avg_similarity: float
    description: str


@dataclass
class QualityMetrics:
    """Quality metrics for search results."""

    relevance_score: float  # 0-1, how relevant results are
    precision_score: float  # 0-1, precision of results
    recall_estimate: float  # 0-1, estimated recall
    keyword_coverage: float  # 0-1, how well keywords are covered
    diversity_score: float  # 0-1, diversity of results
    language_accuracy: float  # 0-1, language filter accuracy


class SearchQualityAnalyzer:
    """Analyze search result quality and relevance."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.search_engine = None
        self.quality_test_cases = self._create_quality_test_cases()

    def _create_quality_test_cases(self) -> list[QualityTestCase]:
        """Create comprehensive quality test cases."""
        return [
            QualityTestCase(
                query="function definition",
                expected_keywords=["def", "function", "return"],
                expected_file_patterns=["*.py", "*.js", "*.ts"],
                expected_languages=["python", "javascript", "typescript"],
                min_results=5,
                max_results=50,
                min_avg_similarity=0.3,
                description="Should find function definitions across languages",
            ),
            QualityTestCase(
                query="class inheritance",
                expected_keywords=["class", "inherit", "extends", "super"],
                expected_file_patterns=["*.py", "*.js", "*.ts"],
                expected_languages=["python", "javascript", "typescript"],
                min_results=2,
                max_results=30,
                min_avg_similarity=0.25,
                description="Should find class inheritance patterns",
            ),
            QualityTestCase(
                query="error handling exception",
                expected_keywords=["try", "catch", "except", "error", "exception"],
                expected_file_patterns=["*.py", "*.js", "*.ts"],
                expected_languages=["python", "javascript", "typescript"],
                min_results=3,
                max_results=40,
                min_avg_similarity=0.3,
                description="Should find error handling code",
            ),
            QualityTestCase(
                query="async await promise",
                expected_keywords=["async", "await", "promise", "asyncio"],
                expected_file_patterns=["*.py", "*.js", "*.ts"],
                expected_languages=["python", "javascript", "typescript"],
                min_results=2,
                max_results=35,
                min_avg_similarity=0.35,
                description="Should find asynchronous code patterns",
            ),
            QualityTestCase(
                query="database connection",
                expected_keywords=["database", "db", "connection", "connect", "client"],
                expected_file_patterns=["*.py", "*.js", "*.ts"],
                expected_languages=["python", "javascript", "typescript"],
                min_results=1,
                max_results=25,
                min_avg_similarity=0.25,
                description="Should find database-related code",
            ),
            QualityTestCase(
                query="test unit testing",
                expected_keywords=["test", "assert", "expect", "mock", "pytest"],
                expected_file_patterns=["test_*.py", "*_test.py", "*.test.js"],
                expected_languages=["python", "javascript", "typescript"],
                min_results=1,
                max_results=30,
                min_avg_similarity=0.3,
                description="Should find test code",
            ),
            QualityTestCase(
                query="configuration settings",
                expected_keywords=[
                    "config",
                    "settings",
                    "options",
                    "env",
                    "environment",
                ],
                expected_file_patterns=["*.py", "*.js", "*.json", "*.yaml"],
                expected_languages=["python", "javascript"],
                min_results=1,
                max_results=20,
                min_avg_similarity=0.25,
                description="Should find configuration code",
            ),
            QualityTestCase(
                query="command line interface CLI",
                expected_keywords=["cli", "command", "argument", "parser", "argparse"],
                expected_file_patterns=["*.py", "*.js"],
                expected_languages=["python", "javascript"],
                min_results=1,
                max_results=15,
                min_avg_similarity=0.3,
                description="Should find CLI-related code",
            ),
        ]

    async def setup_search_engine(self) -> None:
        """Set up the search engine for testing."""
        print("🔧 Setting up search engine...")

        # Initialize project
        project_manager = ProjectManager(self.project_root)

        if not project_manager.is_initialized():
            config = project_manager.initialize(
                file_extensions=[".py", ".js", ".ts", ".md", ".json", ".yaml"],
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                similarity_threshold=0.2,
                force=True,
            )
        else:
            config = project_manager.load_config()

        # Create components
        embedding_function, _ = create_embedding_function(config.embedding_model)
        database = ChromaVectorDatabase(
            persist_directory=config.index_path,
            embedding_function=embedding_function,
        )

        indexer = SemanticIndexer(
            database=database,
            project_root=self.project_root,
            file_extensions=config.file_extensions,
        )

        self.search_engine = SemanticSearchEngine(
            database=database,
            project_root=self.project_root,
            similarity_threshold=config.similarity_threshold,
        )

        # Ensure project is indexed
        await database.initialize()
        indexed_count = await indexer.index_project()
        print(f"  ✓ Indexed {indexed_count} files")

        # Keep database open for search operations

    async def analyze_search_quality(self) -> dict[str, Any]:
        """Analyze overall search quality."""
        print("\n🔍 Analyzing search quality...")

        results = {}
        total_score = 0.0

        for i, test_case in enumerate(self.quality_test_cases, 1):
            print(
                f"\n  Test {i}/{len(self.quality_test_cases)}: {test_case.description}"
            )

            # Perform search
            search_results = await self.search_engine.search(
                query=test_case.query,
                limit=50,
                similarity_threshold=0.1,  # Low threshold to get more results
            )

            # Analyze quality
            metrics = self._analyze_result_quality(search_results, test_case)
            results[test_case.query] = {
                "test_case": test_case,
                "results": search_results,
                "metrics": metrics,
            }

            # Calculate overall score for this test
            test_score = (
                metrics.relevance_score * 0.3
                + metrics.precision_score * 0.25
                + metrics.keyword_coverage * 0.2
                + metrics.diversity_score * 0.15
                + metrics.language_accuracy * 0.1
            )

            total_score += test_score

            print(f"    Results: {len(search_results)}")
            print(f"    Relevance: {metrics.relevance_score:.3f}")
            print(f"    Precision: {metrics.precision_score:.3f}")
            print(f"    Keyword Coverage: {metrics.keyword_coverage:.3f}")
            print(f"    Diversity: {metrics.diversity_score:.3f}")
            print(f"    Test Score: {test_score:.3f}")

        # Calculate overall quality score
        overall_score = total_score / len(self.quality_test_cases)
        results["overall_score"] = overall_score

        return results

    def _analyze_result_quality(
        self, results: list[SearchResult], test_case: QualityTestCase
    ) -> QualityMetrics:
        """Analyze the quality of search results for a test case."""
        if not results:
            return QualityMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # Relevance score based on similarity scores
        avg_similarity = sum(r.similarity_score for r in results) / len(results)
        relevance_score = min(avg_similarity / test_case.min_avg_similarity, 1.0)

        # Precision score based on result count expectations
        result_count = len(results)
        if test_case.min_results <= result_count <= test_case.max_results:
            precision_score = 1.0
        elif result_count < test_case.min_results:
            precision_score = result_count / test_case.min_results
        else:
            precision_score = test_case.max_results / result_count

        # Keyword coverage
        keyword_coverage = self._calculate_keyword_coverage(
            results, test_case.expected_keywords
        )

        # Diversity score
        diversity_score = self._calculate_diversity_score(results)

        # Language accuracy
        language_accuracy = self._calculate_language_accuracy(
            results, test_case.expected_languages
        )

        # Recall estimate (simplified)
        recall_estimate = min(result_count / test_case.max_results, 1.0)

        return QualityMetrics(
            relevance_score=relevance_score,
            precision_score=precision_score,
            recall_estimate=recall_estimate,
            keyword_coverage=keyword_coverage,
            diversity_score=diversity_score,
            language_accuracy=language_accuracy,
        )

    def _calculate_keyword_coverage(
        self, results: list[SearchResult], expected_keywords: list[str]
    ) -> float:
        """Calculate how well expected keywords are covered in results."""
        if not expected_keywords:
            return 1.0

        # Combine all result content
        all_content = " ".join(r.content.lower() for r in results)

        # Count keyword matches
        matches = 0
        for keyword in expected_keywords:
            if keyword.lower() in all_content:
                matches += 1

        return matches / len(expected_keywords)

    def _calculate_diversity_score(self, results: list[SearchResult]) -> float:
        """Calculate diversity of search results."""
        if len(results) <= 1:
            return 1.0

        # File diversity
        unique_files = len({r.file_path for r in results})
        file_diversity = min(unique_files / len(results), 1.0)

        # Language diversity
        unique_languages = len({r.language for r in results})
        max_expected_languages = 3  # Reasonable expectation
        language_diversity = min(unique_languages / max_expected_languages, 1.0)

        # Chunk type diversity
        unique_types = len({r.chunk_type for r in results if r.chunk_type})
        max_expected_types = 4  # function, class, method, code
        type_diversity = (
            min(unique_types / max_expected_types, 1.0) if unique_types > 0 else 0.5
        )

        # Combined diversity score
        return file_diversity * 0.5 + language_diversity * 0.3 + type_diversity * 0.2

    def _calculate_language_accuracy(
        self, results: list[SearchResult], expected_languages: list[str]
    ) -> float:
        """Calculate accuracy of language filtering."""
        if not results:
            return 0.0

        if not expected_languages:
            return 1.0

        # Count results in expected languages
        expected_lang_set = {lang.lower() for lang in expected_languages}
        matching_results = sum(
            1 for r in results if r.language and r.language.lower() in expected_lang_set
        )

        return matching_results / len(results)

    async def analyze_semantic_understanding(self) -> dict[str, Any]:
        """Analyze semantic understanding capabilities."""
        print("\n🧠 Analyzing semantic understanding...")

        semantic_tests = [
            (
                "authentication login security",
                ["auth", "login", "password", "security"],
            ),
            ("data persistence storage", ["database", "save", "store", "persist"]),
            ("user interface frontend", ["ui", "view", "component", "render"]),
            (
                "algorithm optimization performance",
                ["optimize", "efficient", "fast", "performance"],
            ),
            ("network communication protocol", ["http", "request", "response", "api"]),
        ]

        results = {}

        for query, related_concepts in semantic_tests:
            print(f"  Testing: '{query}'")

            search_results = await self.search_engine.search(
                query=query,
                limit=20,
                similarity_threshold=0.2,
            )

            # Analyze semantic understanding
            concept_coverage = self._analyze_concept_coverage(
                search_results, related_concepts
            )
            semantic_coherence = self._analyze_semantic_coherence(search_results, query)

            results[query] = {
                "concept_coverage": concept_coverage,
                "semantic_coherence": semantic_coherence,
                "result_count": len(search_results),
            }

            print(f"    Concept coverage: {concept_coverage:.3f}")
            print(f"    Semantic coherence: {semantic_coherence:.3f}")

        return results

    def _analyze_concept_coverage(
        self, results: list[SearchResult], concepts: list[str]
    ) -> float:
        """Analyze how well related concepts are covered."""
        if not results or not concepts:
            return 0.0

        all_content = " ".join(r.content.lower() for r in results)

        covered_concepts = 0
        for concept in concepts:
            if concept.lower() in all_content:
                covered_concepts += 1

        return covered_concepts / len(concepts)

    def _analyze_semantic_coherence(
        self, results: list[SearchResult], query: str
    ) -> float:
        """Analyze semantic coherence of results."""
        if not results:
            return 0.0

        # Simple coherence measure based on similarity score distribution
        similarities = [r.similarity_score for r in results]

        if len(similarities) <= 1:
            return similarities[0] if similarities else 0.0

        # Check for consistent high-quality results
        high_quality_results = sum(1 for s in similarities if s > 0.5)
        coherence_score = high_quality_results / len(similarities)

        return coherence_score

    async def test_edge_cases(self) -> dict[str, Any]:
        """Test edge cases and robustness."""
        print("\n🧪 Testing edge cases...")

        edge_cases = [
            ("", "Empty query"),
            ("a", "Single character"),
            (
                "very specific nonexistent function name xyz123",
                "Non-existent specific term",
            ),
            ("🚀 emoji query 💻", "Unicode and emoji"),
            ("SELECT * FROM users WHERE id = 1", "SQL injection attempt"),
            ("../../../etc/passwd", "Path traversal"),
            ("x" * 100, "Very long query"),
            ("CamelCaseFunction", "CamelCase query"),
            ("snake_case_function", "Snake case query"),
            ("kebab-case-function", "Kebab case query"),
        ]

        results = {}

        for query, description in edge_cases:
            print(f"  Testing: {description}")

            try:
                search_results = await self.search_engine.search(
                    query=query,
                    limit=10,
                    similarity_threshold=0.1,
                )

                results[description] = {
                    "query": query,
                    "success": True,
                    "result_count": len(search_results),
                    "error": None,
                }

                print(f"    ✓ Success: {len(search_results)} results")

            except Exception as e:
                results[description] = {
                    "query": query,
                    "success": False,
                    "result_count": 0,
                    "error": str(e),
                }

                print(f"    ❌ Error: {e}")

        return results

    def generate_quality_report(
        self,
        quality_results: dict[str, Any],
        semantic_results: dict[str, Any],
        edge_case_results: dict[str, Any],
    ) -> None:
        """Generate comprehensive quality report."""
        print("\n" + "=" * 60)
        print("📊 SEARCH QUALITY ANALYSIS REPORT")
        print("=" * 60)

        # Overall quality score
        overall_score = quality_results.get("overall_score", 0.0)
        print(f"\n🎯 Overall Quality Score: {overall_score:.3f}/1.000")

        if overall_score >= 0.8:
            print("  ✅ Excellent search quality")
        elif overall_score >= 0.6:
            print("  ✅ Good search quality")
        elif overall_score >= 0.4:
            print("  ⚠️  Acceptable search quality")
        else:
            print("  ❌ Poor search quality - needs improvement")

        # Quality test breakdown
        print("\n📋 Quality Test Results:")
        for query, result in quality_results.items():
            if query == "overall_score":
                continue

            metrics = result["metrics"]
            print(f"\n  Query: '{query}'")
            print(f"    Relevance: {metrics.relevance_score:.3f}")
            print(f"    Precision: {metrics.precision_score:.3f}")
            print(f"    Keyword Coverage: {metrics.keyword_coverage:.3f}")
            print(f"    Diversity: {metrics.diversity_score:.3f}")
            print(f"    Language Accuracy: {metrics.language_accuracy:.3f}")

        # Semantic understanding
        print("\n🧠 Semantic Understanding:")
        semantic_scores = []
        for query, result in semantic_results.items():
            concept_score = result["concept_coverage"]
            coherence_score = result["semantic_coherence"]
            combined_score = (concept_score + coherence_score) / 2
            semantic_scores.append(combined_score)

            print(
                f"  '{query}': {combined_score:.3f} (concept: {concept_score:.3f}, coherence: {coherence_score:.3f})"
            )

        avg_semantic_score = (
            sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0.0
        )
        print(f"  Average semantic score: {avg_semantic_score:.3f}")

        # Edge case robustness
        print("\n🛡️  Edge Case Robustness:")
        successful_edge_cases = sum(
            1 for r in edge_case_results.values() if r["success"]
        )
        total_edge_cases = len(edge_case_results)
        robustness_score = successful_edge_cases / total_edge_cases

        print(
            f"  Successful edge cases: {successful_edge_cases}/{total_edge_cases} ({robustness_score:.1%})"
        )

        for description, result in edge_case_results.items():
            status = "✓" if result["success"] else "❌"
            print(f"    {status} {description}")

        # Recommendations
        print("\n💡 Recommendations:")

        if overall_score < 0.6:
            print("  • Review and improve embedding model quality")
            print("  • Optimize similarity threshold settings")
            print("  • Enhance query preprocessing and expansion")

        if avg_semantic_score < 0.5:
            print("  • Improve semantic understanding with better embeddings")
            print("  • Add query expansion for related concepts")
            print("  • Consider fine-tuning embedding model on code data")

        if robustness_score < 0.8:
            print("  • Improve error handling for edge cases")
            print("  • Add input validation and sanitization")
            print("  • Handle unicode and special characters better")

        # Performance recommendations
        avg_result_counts = [
            len(r["results"])
            for r in quality_results.values()
            if isinstance(r, dict) and "results" in r
        ]
        if avg_result_counts:
            avg_results = sum(avg_result_counts) / len(avg_result_counts)
            if avg_results < 5:
                print("  • Consider lowering similarity threshold for more results")
            elif avg_results > 30:
                print(
                    "  • Consider raising similarity threshold for more precise results"
                )

        print("\n" + "=" * 60)


async def main():
    """Main analysis execution."""
    print("🔍 MCP Vector Search - Quality Analysis")
    print("=" * 50)

    project_root = Path.cwd()
    analyzer = SearchQualityAnalyzer(project_root)

    try:
        # Setup
        await analyzer.setup_search_engine()

        # Run analyses
        quality_results = await analyzer.analyze_search_quality()
        semantic_results = await analyzer.analyze_semantic_understanding()
        edge_case_results = await analyzer.test_edge_cases()

        # Generate report
        analyzer.generate_quality_report(
            quality_results, semantic_results, edge_case_results
        )

        print("\n🎉 Quality analysis completed!")

    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
