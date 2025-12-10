"""Unit tests for metric dataclasses."""

from datetime import datetime

from mcp_vector_search.analysis import ChunkMetrics, FileMetrics, ProjectMetrics


class TestChunkMetrics:
    """Test ChunkMetrics dataclass."""

    def test_default_values(self):
        """Test default metric values."""
        metrics = ChunkMetrics()
        assert metrics.cognitive_complexity == 0
        assert metrics.cyclomatic_complexity == 0
        assert metrics.max_nesting_depth == 0
        assert metrics.parameter_count == 0
        assert metrics.lines_of_code == 0
        assert metrics.smells == []
        assert metrics.complexity_grade == "A"

    def test_grade_calculation_a(self):
        """Test A grade boundary (0-5)."""
        assert ChunkMetrics(cognitive_complexity=0).complexity_grade == "A"
        assert ChunkMetrics(cognitive_complexity=3).complexity_grade == "A"
        assert ChunkMetrics(cognitive_complexity=5).complexity_grade == "A"

    def test_grade_calculation_b(self):
        """Test B grade boundary (6-10)."""
        assert ChunkMetrics(cognitive_complexity=6).complexity_grade == "B"
        assert ChunkMetrics(cognitive_complexity=8).complexity_grade == "B"
        assert ChunkMetrics(cognitive_complexity=10).complexity_grade == "B"

    def test_grade_calculation_c(self):
        """Test C grade boundary (11-20)."""
        assert ChunkMetrics(cognitive_complexity=11).complexity_grade == "C"
        assert ChunkMetrics(cognitive_complexity=15).complexity_grade == "C"
        assert ChunkMetrics(cognitive_complexity=20).complexity_grade == "C"

    def test_grade_calculation_d(self):
        """Test D grade boundary (21-30)."""
        assert ChunkMetrics(cognitive_complexity=21).complexity_grade == "D"
        assert ChunkMetrics(cognitive_complexity=25).complexity_grade == "D"
        assert ChunkMetrics(cognitive_complexity=30).complexity_grade == "D"

    def test_grade_calculation_f(self):
        """Test F grade boundary (31+)."""
        assert ChunkMetrics(cognitive_complexity=31).complexity_grade == "F"
        assert ChunkMetrics(cognitive_complexity=50).complexity_grade == "F"
        assert ChunkMetrics(cognitive_complexity=100).complexity_grade == "F"

    def test_to_metadata_basic(self):
        """Test ChromaDB metadata flattening with basic metrics."""
        metrics = ChunkMetrics(
            cognitive_complexity=15,
            cyclomatic_complexity=10,
            max_nesting_depth=3,
            parameter_count=4,
            lines_of_code=50,
        )
        metadata = metrics.to_metadata()

        # Verify all expected fields present
        assert "cognitive_complexity" in metadata
        assert "cyclomatic_complexity" in metadata
        assert "max_nesting_depth" in metadata
        assert "parameter_count" in metadata
        assert "lines_of_code" in metadata
        assert "complexity_grade" in metadata
        assert "code_smells" in metadata
        assert "smell_count" in metadata

        # Verify values
        assert metadata["cognitive_complexity"] == 15
        assert metadata["cyclomatic_complexity"] == 10
        assert metadata["max_nesting_depth"] == 3
        assert metadata["parameter_count"] == 4
        assert metadata["lines_of_code"] == 50
        assert metadata["complexity_grade"] == "C"
        assert metadata["code_smells"] == "[]"  # JSON string for ChromaDB compatibility
        assert metadata["smell_count"] == 0

    def test_to_metadata_with_smells(self):
        """Test ChromaDB metadata with code smells."""
        metrics = ChunkMetrics(
            cognitive_complexity=15,
            cyclomatic_complexity=10,
            smells=["long_method", "too_many_params"],
        )
        metadata = metrics.to_metadata()

        # Verify all values are ChromaDB-compatible types
        assert isinstance(metadata["cognitive_complexity"], int)
        assert isinstance(metadata["complexity_grade"], str)
        assert isinstance(metadata["code_smells"], str)  # JSON string for ChromaDB
        assert isinstance(metadata["smell_count"], int)

        # Verify smell data (stored as JSON string)
        import json

        smells = json.loads(metadata["code_smells"])
        assert "long_method" in smells
        assert "too_many_params" in smells
        assert metadata["smell_count"] == 2

    def test_to_metadata_chromadb_compatibility(self):
        """Test all metadata values are ChromaDB-compatible types."""
        metrics = ChunkMetrics(
            cognitive_complexity=8,
            cyclomatic_complexity=5,
            max_nesting_depth=2,
            parameter_count=3,
            lines_of_code=30,
            smells=["test_smell"],
        )
        metadata = metrics.to_metadata()

        # ChromaDB supports: str, int, float, bool, list[str]
        for key, value in metadata.items():
            assert isinstance(value, str | int | float | bool | list), (
                f"Invalid type for {key}: {type(value)}"
            )

            # If list, verify all elements are strings
            if isinstance(value, list):
                assert all(isinstance(item, str) for item in value), (
                    f"List {key} contains non-string elements"
                )


class TestFileMetrics:
    """Test FileMetrics dataclass."""

    def test_default_values(self):
        """Test default file metric values."""
        metrics = FileMetrics(file_path="test.py")
        assert metrics.file_path == "test.py"
        assert metrics.total_lines == 0
        assert metrics.code_lines == 0
        assert metrics.comment_lines == 0
        assert metrics.blank_lines == 0
        assert metrics.function_count == 0
        assert metrics.class_count == 0
        assert metrics.method_count == 0
        assert metrics.total_complexity == 0
        assert metrics.avg_complexity == 0.0
        assert metrics.max_complexity == 0
        assert metrics.chunks == []

    def test_compute_aggregates_no_chunks(self):
        """Test aggregate computation with no chunks."""
        metrics = FileMetrics(file_path="test.py")
        metrics.compute_aggregates()

        assert metrics.total_complexity == 0
        assert metrics.avg_complexity == 0.0
        assert metrics.max_complexity == 0

    def test_compute_aggregates_single_chunk(self):
        """Test aggregate computation with single chunk."""
        chunk = ChunkMetrics(cognitive_complexity=10)
        metrics = FileMetrics(file_path="test.py", chunks=[chunk])
        metrics.compute_aggregates()

        assert metrics.total_complexity == 10
        assert metrics.avg_complexity == 10.0
        assert metrics.max_complexity == 10

    def test_compute_aggregates_multiple_chunks(self):
        """Test aggregate computation with multiple chunks."""
        chunks = [
            ChunkMetrics(cognitive_complexity=5),
            ChunkMetrics(cognitive_complexity=10),
            ChunkMetrics(cognitive_complexity=15),
            ChunkMetrics(cognitive_complexity=20),
        ]
        metrics = FileMetrics(file_path="test.py", chunks=chunks)
        metrics.compute_aggregates()

        assert metrics.total_complexity == 50  # 5+10+15+20
        assert metrics.avg_complexity == 12.5  # 50/4
        assert metrics.max_complexity == 20

    def test_health_score_perfect(self):
        """Test health score with perfect metrics."""
        metrics = FileMetrics(
            file_path="test.py",
            total_lines=100,
            code_lines=70,
            comment_lines=20,
            blank_lines=10,
            chunks=[ChunkMetrics(cognitive_complexity=3)],
        )
        metrics.compute_aggregates()

        health = metrics.health_score
        assert 0.0 <= health <= 1.0
        assert health >= 0.9  # Should be near perfect

    def test_health_score_high_complexity(self):
        """Test health score penalty for high complexity."""
        # Test F grade complexity (>30)
        chunks = [ChunkMetrics(cognitive_complexity=40)]
        metrics = FileMetrics(file_path="test.py", chunks=chunks)
        metrics.compute_aggregates()

        health_f = metrics.health_score
        assert health_f <= 0.5  # -50% penalty

        # Test D grade complexity (21-30)
        chunks = [ChunkMetrics(cognitive_complexity=25)]
        metrics = FileMetrics(file_path="test.py", chunks=chunks)
        metrics.compute_aggregates()

        health_d = metrics.health_score
        assert health_d <= 0.7  # -30% penalty

    def test_health_score_code_smells(self):
        """Test health score penalty for code smells."""
        chunks = [
            ChunkMetrics(cognitive_complexity=5, smells=["smell1", "smell2", "smell3"])
        ]
        metrics = FileMetrics(file_path="test.py", chunks=chunks)
        metrics.compute_aggregates()

        health = metrics.health_score
        # 3 smells = 15% penalty (5% per smell)
        assert health <= 0.85

    def test_health_score_comment_ratio(self):
        """Test health score with various comment ratios."""
        # Too few comments (<10%)
        metrics_low = FileMetrics(
            file_path="test.py",
            total_lines=100,
            code_lines=95,
            comment_lines=5,
            chunks=[ChunkMetrics(cognitive_complexity=3)],
        )
        metrics_low.compute_aggregates()

        # Good comment ratio (10-30%)
        metrics_good = FileMetrics(
            file_path="test.py",
            total_lines=100,
            code_lines=75,
            comment_lines=20,
            chunks=[ChunkMetrics(cognitive_complexity=3)],
        )
        metrics_good.compute_aggregates()

        # Too many comments (>50%)
        metrics_high = FileMetrics(
            file_path="test.py",
            total_lines=100,
            code_lines=40,
            comment_lines=55,
            chunks=[ChunkMetrics(cognitive_complexity=3)],
        )
        metrics_high.compute_aggregates()

        assert metrics_good.health_score > metrics_low.health_score
        assert metrics_good.health_score > metrics_high.health_score

    def test_health_score_bounds(self):
        """Test health score is always between 0.0 and 1.0."""
        # Create worst-case metrics
        chunks = [
            ChunkMetrics(
                cognitive_complexity=100, smells=["s1", "s2", "s3", "s4", "s5", "s6"]
            )
        ]
        metrics = FileMetrics(
            file_path="test.py",
            total_lines=100,
            code_lines=99,
            comment_lines=0,
            chunks=chunks,
        )
        metrics.compute_aggregates()

        health = metrics.health_score
        assert 0.0 <= health <= 1.0


class TestProjectMetrics:
    """Test ProjectMetrics dataclass."""

    def test_default_values(self):
        """Test default project metric values."""
        metrics = ProjectMetrics(project_root="/test/project")
        assert metrics.project_root == "/test/project"
        assert isinstance(metrics.analyzed_at, datetime)
        assert metrics.total_files == 0
        assert metrics.total_lines == 0
        assert metrics.total_functions == 0
        assert metrics.total_classes == 0
        assert metrics.files == {}
        assert metrics.avg_file_complexity == 0.0
        assert metrics.hotspots == []

    def test_compute_aggregates_no_files(self):
        """Test aggregate computation with no files."""
        metrics = ProjectMetrics(project_root="/test/project")
        metrics.compute_aggregates()

        assert metrics.total_files == 0
        assert metrics.total_lines == 0
        assert metrics.total_functions == 0
        assert metrics.total_classes == 0
        assert metrics.avg_file_complexity == 0.0
        assert metrics.hotspots == []

    def test_compute_aggregates_single_file(self):
        """Test aggregate computation with single file."""
        file_metrics = FileMetrics(
            file_path="test.py",
            total_lines=100,
            function_count=5,
            class_count=2,
            chunks=[ChunkMetrics(cognitive_complexity=10)],
        )
        file_metrics.compute_aggregates()

        project = ProjectMetrics(project_root="/test/project")
        project.files["test.py"] = file_metrics
        project.compute_aggregates()

        assert project.total_files == 1
        assert project.total_lines == 100
        assert project.total_functions == 5
        assert project.total_classes == 2
        assert project.avg_file_complexity == 10.0
        assert "test.py" in project.hotspots

    def test_compute_aggregates_multiple_files(self):
        """Test aggregate computation with multiple files."""
        file1 = FileMetrics(
            file_path="file1.py",
            total_lines=100,
            function_count=5,
            class_count=2,
            chunks=[ChunkMetrics(cognitive_complexity=10)],
        )
        file1.compute_aggregates()

        file2 = FileMetrics(
            file_path="file2.py",
            total_lines=200,
            function_count=8,
            class_count=3,
            chunks=[ChunkMetrics(cognitive_complexity=20)],
        )
        file2.compute_aggregates()

        project = ProjectMetrics(project_root="/test/project")
        project.files["file1.py"] = file1
        project.files["file2.py"] = file2
        project.compute_aggregates()

        assert project.total_files == 2
        assert project.total_lines == 300  # 100 + 200
        assert project.total_functions == 13  # 5 + 8
        assert project.total_classes == 5  # 2 + 3
        assert project.avg_file_complexity == 15.0  # (10 + 20) / 2

    def test_get_hotspots_ordering(self):
        """Test hotspots are ordered by complexity (highest first)."""
        files = {
            "low.py": FileMetrics(
                file_path="low.py", chunks=[ChunkMetrics(cognitive_complexity=5)]
            ),
            "high.py": FileMetrics(
                file_path="high.py", chunks=[ChunkMetrics(cognitive_complexity=30)]
            ),
            "medium.py": FileMetrics(
                file_path="medium.py", chunks=[ChunkMetrics(cognitive_complexity=15)]
            ),
        }

        for file_metrics in files.values():
            file_metrics.compute_aggregates()

        project = ProjectMetrics(project_root="/test/project")
        project.files = files

        hotspots = project.get_hotspots(limit=10)

        # Should be ordered: high -> medium -> low
        assert len(hotspots) == 3
        assert hotspots[0].file_path == "high.py"
        assert hotspots[1].file_path == "medium.py"
        assert hotspots[2].file_path == "low.py"

    def test_get_hotspots_limit(self):
        """Test hotspots respect limit parameter."""
        files = {}
        for i in range(20):
            file_metrics = FileMetrics(
                file_path=f"file{i}.py", chunks=[ChunkMetrics(cognitive_complexity=i)]
            )
            file_metrics.compute_aggregates()
            files[f"file{i}.py"] = file_metrics

        project = ProjectMetrics(project_root="/test/project")
        project.files = files

        hotspots_5 = project.get_hotspots(limit=5)
        assert len(hotspots_5) == 5

        hotspots_10 = project.get_hotspots(limit=10)
        assert len(hotspots_10) == 10

    def test_get_hotspots_excludes_empty_files(self):
        """Test hotspots exclude files with no chunks."""
        file_with_chunks = FileMetrics(
            file_path="has_chunks.py", chunks=[ChunkMetrics(cognitive_complexity=10)]
        )
        file_with_chunks.compute_aggregates()

        file_no_chunks = FileMetrics(file_path="no_chunks.py", chunks=[])
        file_no_chunks.compute_aggregates()

        project = ProjectMetrics(project_root="/test/project")
        project.files["has_chunks.py"] = file_with_chunks
        project.files["no_chunks.py"] = file_no_chunks

        hotspots = project.get_hotspots(limit=10)

        # Should only include file with chunks
        assert len(hotspots) == 1
        assert hotspots[0].file_path == "has_chunks.py"

    def test_to_summary_structure(self):
        """Test summary dict has expected structure."""
        project = ProjectMetrics(project_root="/test/project")
        summary = project.to_summary()

        # Verify top-level keys
        assert "project_root" in summary
        assert "analyzed_at" in summary
        assert "total_files" in summary
        assert "total_lines" in summary
        assert "total_functions" in summary
        assert "total_classes" in summary
        assert "avg_file_complexity" in summary
        assert "hotspots" in summary
        assert "complexity_distribution" in summary
        assert "health_metrics" in summary

        # Verify nested structures
        assert "avg_health_score" in summary["health_metrics"]
        assert "files_needing_attention" in summary["health_metrics"]

        distribution = summary["complexity_distribution"]
        assert "A" in distribution
        assert "B" in distribution
        assert "C" in distribution
        assert "D" in distribution
        assert "F" in distribution

    def test_to_summary_with_data(self):
        """Test summary with actual data."""
        file_metrics = FileMetrics(
            file_path="test.py",
            total_lines=100,
            function_count=5,
            class_count=2,
            chunks=[
                ChunkMetrics(cognitive_complexity=5),  # A
                ChunkMetrics(cognitive_complexity=10),  # B
                ChunkMetrics(cognitive_complexity=15),  # C
            ],
        )
        file_metrics.compute_aggregates()

        project = ProjectMetrics(project_root="/test/project")
        project.files["test.py"] = file_metrics
        project.compute_aggregates()

        summary = project.to_summary()

        assert summary["project_root"] == "/test/project"
        assert summary["total_files"] == 1
        assert summary["total_lines"] == 100
        assert summary["total_functions"] == 5
        assert summary["total_classes"] == 2

        # Verify complexity distribution
        dist = summary["complexity_distribution"]
        assert dist["A"] == 1
        assert dist["B"] == 1
        assert dist["C"] == 1
        assert dist["D"] == 0
        assert dist["F"] == 0

    def test_grade_distribution(self):
        """Test complexity grade distribution calculation."""
        files = {
            "file1.py": FileMetrics(
                file_path="file1.py",
                chunks=[
                    ChunkMetrics(cognitive_complexity=3),  # A
                    ChunkMetrics(cognitive_complexity=8),  # B
                ],
            ),
            "file2.py": FileMetrics(
                file_path="file2.py",
                chunks=[
                    ChunkMetrics(cognitive_complexity=15),  # C
                    ChunkMetrics(cognitive_complexity=25),  # D
                    ChunkMetrics(cognitive_complexity=40),  # F
                ],
            ),
        }

        for file_metrics in files.values():
            file_metrics.compute_aggregates()

        project = ProjectMetrics(project_root="/test/project")
        project.files = files
        project.compute_aggregates()

        distribution = project._compute_grade_distribution()

        assert distribution["A"] == 1
        assert distribution["B"] == 1
        assert distribution["C"] == 1
        assert distribution["D"] == 1
        assert distribution["F"] == 1

    def test_avg_health_score(self):
        """Test average health score calculation."""
        # Create files with different health scores
        file1 = FileMetrics(
            file_path="file1.py",
            total_lines=100,
            comment_lines=20,
            chunks=[ChunkMetrics(cognitive_complexity=5)],  # Good
        )
        file1.compute_aggregates()

        file2 = FileMetrics(
            file_path="file2.py",
            total_lines=100,
            comment_lines=0,
            chunks=[ChunkMetrics(cognitive_complexity=40)],  # Poor
        )
        file2.compute_aggregates()

        project = ProjectMetrics(project_root="/test/project")
        project.files["file1.py"] = file1
        project.files["file2.py"] = file2

        avg_health = project._compute_avg_health_score()

        # Should be average of two health scores
        expected_avg = (file1.health_score + file2.health_score) / 2
        assert abs(avg_health - expected_avg) < 0.001  # Float comparison

    def test_files_needing_attention(self):
        """Test counting files with low health scores."""
        # Create files with different health scores
        good_file = FileMetrics(
            file_path="good.py",
            total_lines=100,
            comment_lines=20,
            chunks=[ChunkMetrics(cognitive_complexity=5)],
        )
        good_file.compute_aggregates()

        bad_file1 = FileMetrics(
            file_path="bad1.py",
            total_lines=100,
            comment_lines=0,
            chunks=[ChunkMetrics(cognitive_complexity=40, smells=["s1", "s2"])],
        )
        bad_file1.compute_aggregates()

        bad_file2 = FileMetrics(
            file_path="bad2.py",
            total_lines=100,
            comment_lines=0,
            chunks=[ChunkMetrics(cognitive_complexity=35)],
        )
        bad_file2.compute_aggregates()

        project = ProjectMetrics(project_root="/test/project")
        project.files["good.py"] = good_file
        project.files["bad1.py"] = bad_file1
        project.files["bad2.py"] = bad_file2

        # Count files with health < 0.7
        needs_attention = project._count_files_needing_attention()

        # Both bad files should have health < 0.7
        assert needs_attention >= 2
