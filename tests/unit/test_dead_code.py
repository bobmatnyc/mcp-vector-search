"""Unit tests for dead code detection."""

from pathlib import Path

from src.mcp_vector_search.analysis.dead_code import (
    Confidence,
    DeadCodeAnalyzer,
    DeadCodeFinding,
    DeadCodeReport,
)
from src.mcp_vector_search.analysis.entry_points import EntryPoint, EntryPointType


class TestConfidence:
    """Test Confidence enum."""

    def test_confidence_levels(self):
        """Test that all confidence levels exist."""
        assert Confidence.HIGH.value == "HIGH"
        assert Confidence.MEDIUM.value == "MEDIUM"
        assert Confidence.LOW.value == "LOW"

    def test_string_conversion(self):
        """Test string conversion."""
        assert str(Confidence.HIGH) == "HIGH"
        assert str(Confidence.MEDIUM) == "MEDIUM"
        assert str(Confidence.LOW) == "LOW"


class TestDeadCodeFinding:
    """Test DeadCodeFinding dataclass."""

    def test_creation(self):
        """Test creating a finding."""
        finding = DeadCodeFinding(
            function_name="unused_function",
            file_path="module.py",
            start_line=10,
            end_line=20,
            confidence=Confidence.HIGH,
            reason="Not called from any entry point",
            caveats=["Might be called dynamically"],
        )

        assert finding.function_name == "unused_function"
        assert finding.file_path == "module.py"
        assert finding.start_line == 10
        assert finding.end_line == 20
        assert finding.confidence == Confidence.HIGH
        assert finding.reason == "Not called from any entry point"
        assert len(finding.caveats) == 1

    def test_string_representation(self):
        """Test string representation."""
        finding = DeadCodeFinding(
            function_name="unused",
            file_path="test.py",
            start_line=5,
            end_line=10,
            confidence=Confidence.MEDIUM,
            reason="Test reason",
        )

        finding_str = str(finding)
        assert "MEDIUM" in finding_str
        assert "unused" in finding_str
        assert "test.py:5-10" in finding_str


class TestDeadCodeReport:
    """Test DeadCodeReport dataclass."""

    def test_creation(self):
        """Test creating a report."""
        report = DeadCodeReport(
            entry_points=[],
            findings=[],
            total_functions=100,
            reachable_count=80,
            unreachable_count=20,
        )

        assert report.total_functions == 100
        assert report.reachable_count == 80
        assert report.unreachable_count == 20

    def test_reachable_percentage(self):
        """Test calculation of reachable percentage."""
        report = DeadCodeReport(
            entry_points=[],
            findings=[],
            total_functions=100,
            reachable_count=80,
            unreachable_count=20,
        )

        assert report.reachable_percentage == 80.0

    def test_reachable_percentage_zero_functions(self):
        """Test reachable percentage with zero functions."""
        report = DeadCodeReport(
            entry_points=[],
            findings=[],
            total_functions=0,
            reachable_count=0,
            unreachable_count=0,
        )

        # Should return 100% for empty project
        assert report.reachable_percentage == 100.0

    def test_get_findings_by_confidence(self):
        """Test filtering findings by confidence."""
        findings = [
            DeadCodeFinding(
                function_name="f1",
                file_path="a.py",
                start_line=1,
                end_line=5,
                confidence=Confidence.HIGH,
                reason="Test",
            ),
            DeadCodeFinding(
                function_name="f2",
                file_path="b.py",
                start_line=1,
                end_line=5,
                confidence=Confidence.MEDIUM,
                reason="Test",
            ),
            DeadCodeFinding(
                function_name="f3",
                file_path="c.py",
                start_line=1,
                end_line=5,
                confidence=Confidence.HIGH,
                reason="Test",
            ),
        ]

        report = DeadCodeReport(
            entry_points=[],
            findings=findings,
            total_functions=3,
            reachable_count=0,
            unreachable_count=3,
        )

        high_findings = report.get_findings_by_confidence(Confidence.HIGH)
        assert len(high_findings) == 2

        medium_findings = report.get_findings_by_confidence(Confidence.MEDIUM)
        assert len(medium_findings) == 1


class TestDeadCodeAnalyzer:
    """Test DeadCodeAnalyzer class."""

    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = DeadCodeAnalyzer(
            include_public_entry_points=False,
            custom_entry_points=["custom_func"],
            exclude_patterns=["test_"],
            min_confidence=Confidence.MEDIUM,
        )

        assert analyzer.include_public is False
        assert analyzer.custom_entry_points == ["custom_func"]
        assert analyzer.exclude_patterns == ["test_"]
        assert analyzer.min_confidence == Confidence.MEDIUM

    def test_build_call_graph_simple(self):
        """Test building call graph from chunks."""
        chunks = [
            {
                "type": "function",
                "function_name": "main",
                "content": "def main():\n    helper1()\n    helper2()",
            },
            {
                "type": "function",
                "function_name": "helper1",
                "content": "def helper1():\n    pass",
            },
            {
                "type": "function",
                "function_name": "helper2",
                "content": "def helper2():\n    helper1()",
            },
        ]

        analyzer = DeadCodeAnalyzer()
        call_graph = analyzer._build_call_graph(chunks)

        # main calls helper1 and helper2
        assert "helper1" in call_graph["main"]
        assert "helper2" in call_graph["main"]

        # helper2 calls helper1
        assert "helper1" in call_graph["helper2"]

        # helper1 calls nothing
        assert len(call_graph.get("helper1", set())) == 0

    def test_build_call_graph_skips_non_functions(self):
        """Test that call graph only includes functions."""
        chunks = [
            {
                "type": "function",
                "function_name": "func",
                "content": "def func(): pass",
            },
            {
                "type": "class",
                "class_name": "MyClass",
                "content": "class MyClass: pass",
            },
            {
                "type": "comment",
                "content": "# Just a comment",
            },
        ]

        analyzer = DeadCodeAnalyzer()
        call_graph = analyzer._build_call_graph(chunks)

        # Should include function but not class or comment
        assert "func" in call_graph

    def test_compute_reachability_simple(self):
        """Test computing reachability from entry points."""
        entry_points = [
            EntryPoint(
                name="main",
                file_path="main.py",
                line_number=1,
                type=EntryPointType.MAIN,
            )
        ]

        call_graph = {
            "main": {"helper1", "helper2"},
            "helper1": {"helper3"},
            "helper2": set(),
            "helper3": set(),
            "unused": set(),  # Not reachable from main
        }

        chunks = []  # Not needed for this test

        analyzer = DeadCodeAnalyzer()
        reachable = analyzer._compute_reachability(entry_points, call_graph, chunks)

        # Should reach main, helper1, helper2, helper3
        assert "main" in reachable
        assert "helper1" in reachable
        assert "helper2" in reachable
        assert "helper3" in reachable

        # Should NOT reach unused
        assert "unused" not in reachable

    def test_compute_reachability_multiple_entry_points(self):
        """Test reachability with multiple entry points."""
        entry_points = [
            EntryPoint(
                name="main",
                file_path="main.py",
                line_number=1,
                type=EntryPointType.MAIN,
            ),
            EntryPoint(
                name="test_func",
                file_path="test.py",
                line_number=1,
                type=EntryPointType.TEST,
            ),
        ]

        call_graph = {
            "main": {"helper1"},
            "test_func": {"helper2"},
            "helper1": set(),
            "helper2": set(),
            "unused": set(),
        }

        chunks = []

        analyzer = DeadCodeAnalyzer()
        reachable = analyzer._compute_reachability(entry_points, call_graph, chunks)

        # Should reach both entry points and their callees
        assert "main" in reachable
        assert "test_func" in reachable
        assert "helper1" in reachable
        assert "helper2" in reachable
        assert "unused" not in reachable

    def test_find_dead_code_simple(self):
        """Test finding dead code in simple case."""
        chunks = [
            {
                "type": "function",
                "function_name": "used",
                "file_path": "module.py",
                "start_line": 1,
                "end_line": 5,
                "decorators": [],
            },
            {
                "type": "function",
                "function_name": "unused",
                "file_path": "module.py",
                "start_line": 7,
                "end_line": 12,
                "decorators": [],
            },
        ]

        reachable = {"used"}

        analyzer = DeadCodeAnalyzer()
        findings = analyzer._find_dead_code(chunks, reachable)

        assert len(findings) == 1
        assert findings[0].function_name == "unused"

    def test_find_dead_code_skips_reachable(self):
        """Test that reachable functions are not reported."""
        chunks = [
            {
                "type": "function",
                "function_name": "func1",
                "file_path": "module.py",
                "start_line": 1,
                "end_line": 5,
                "decorators": [],
            },
            {
                "type": "function",
                "function_name": "func2",
                "file_path": "module.py",
                "start_line": 7,
                "end_line": 12,
                "decorators": [],
            },
        ]

        reachable = {"func1", "func2"}  # Both reachable

        analyzer = DeadCodeAnalyzer()
        findings = analyzer._find_dead_code(chunks, reachable)

        assert len(findings) == 0

    def test_assign_confidence_private_function(self):
        """Test confidence assignment for private functions."""
        chunk = {
            "function_name": "_private_func",
            "decorators": [],
        }

        analyzer = DeadCodeAnalyzer()
        confidence = analyzer._assign_confidence(chunk)

        # Private functions get HIGH confidence
        assert confidence == Confidence.HIGH

    def test_assign_confidence_public_function(self):
        """Test confidence assignment for public functions."""
        chunk = {
            "function_name": "public_func",
            "decorators": [],
        }

        analyzer = DeadCodeAnalyzer()
        confidence = analyzer._assign_confidence(chunk)

        # Public functions get MEDIUM confidence
        assert confidence == Confidence.MEDIUM

    def test_assign_confidence_decorated_function(self):
        """Test confidence assignment for decorated functions."""
        chunk = {
            "function_name": "decorated_func",
            "decorators": ["@app.route"],
        }

        analyzer = DeadCodeAnalyzer()
        confidence = analyzer._assign_confidence(chunk)

        # Decorated functions get LOW confidence (might be registered)
        assert confidence == Confidence.LOW

    def test_generate_caveats_for_decorators(self):
        """Test caveat generation for decorated functions."""
        chunk = {
            "function_name": "handler",
            "decorators": ["@app.route", "@login_required"],
        }

        analyzer = DeadCodeAnalyzer()
        caveats = analyzer._generate_caveats(chunk)

        assert len(caveats) > 0
        assert any("decorator" in caveat.lower() for caveat in caveats)

    def test_generate_caveats_for_callback_names(self):
        """Test caveat generation for callback-like names."""
        chunks = [
            {"function_name": "on_click_handler", "decorators": []},
            {"function_name": "handle_request", "decorators": []},
            {"function_name": "my_callback", "decorators": []},
        ]

        analyzer = DeadCodeAnalyzer()

        for chunk in chunks:
            caveats = analyzer._generate_caveats(chunk)
            assert len(caveats) > 0
            assert any(
                "callback" in caveat.lower() or "handler" in caveat.lower()
                for caveat in caveats
            )

    def test_is_excluded(self):
        """Test file path exclusion."""
        analyzer = DeadCodeAnalyzer(exclude_patterns=["test_", "migrations/"])

        assert analyzer._is_excluded("test_module.py") is True
        assert analyzer._is_excluded("src/migrations/0001.py") is True
        assert analyzer._is_excluded("src/module.py") is False

    def test_meets_confidence_high(self):
        """Test confidence threshold check for HIGH."""
        analyzer = DeadCodeAnalyzer(min_confidence=Confidence.HIGH)

        assert analyzer._meets_confidence(Confidence.HIGH) is True
        assert analyzer._meets_confidence(Confidence.MEDIUM) is False
        assert analyzer._meets_confidence(Confidence.LOW) is False

    def test_meets_confidence_medium(self):
        """Test confidence threshold check for MEDIUM."""
        analyzer = DeadCodeAnalyzer(min_confidence=Confidence.MEDIUM)

        assert analyzer._meets_confidence(Confidence.HIGH) is True
        assert analyzer._meets_confidence(Confidence.MEDIUM) is True
        assert analyzer._meets_confidence(Confidence.LOW) is False

    def test_meets_confidence_low(self):
        """Test confidence threshold check for LOW."""
        analyzer = DeadCodeAnalyzer(min_confidence=Confidence.LOW)

        assert analyzer._meets_confidence(Confidence.HIGH) is True
        assert analyzer._meets_confidence(Confidence.MEDIUM) is True
        assert analyzer._meets_confidence(Confidence.LOW) is True

    def test_analyze_integration(self):
        """Test full analysis integration."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create test file with entry point
            (tmppath / "main.py").write_text(
                """
def main():
    used_function()

def used_function():
    pass

def unused_function():
    pass

if __name__ == "__main__":
    main()
"""
            )

            # Create chunks (simulating indexed code)
            chunks = [
                {
                    "type": "function",
                    "function_name": "main",
                    "file_path": str(tmppath / "main.py"),
                    "start_line": 2,
                    "end_line": 3,
                    "content": "def main():\n    used_function()",
                    "decorators": [],
                },
                {
                    "type": "function",
                    "function_name": "used_function",
                    "file_path": str(tmppath / "main.py"),
                    "start_line": 5,
                    "end_line": 6,
                    "content": "def used_function():\n    pass",
                    "decorators": [],
                },
                {
                    "type": "function",
                    "function_name": "unused_function",
                    "file_path": str(tmppath / "main.py"),
                    "start_line": 8,
                    "end_line": 9,
                    "content": "def unused_function():\n    pass",
                    "decorators": [],
                },
            ]

            analyzer = DeadCodeAnalyzer(min_confidence=Confidence.LOW)
            report = analyzer.analyze(tmppath, chunks)

            # Should have 1 entry point (main)
            assert len(report.entry_points) > 0

            # Should have 1 finding (unused_function)
            assert len(report.findings) == 1
            assert report.findings[0].function_name == "unused_function"

            # Reachability stats
            assert report.reachable_count == 2  # main, used_function
            assert report.unreachable_count == 1  # unused_function
            assert report.total_functions == 3

    def test_custom_entry_points(self):
        """Test using custom entry points."""
        chunks = [
            {
                "type": "function",
                "function_name": "custom_entry",
                "file_path": "module.py",
                "start_line": 1,
                "end_line": 5,
                "content": "def custom_entry():\n    helper()",
                "decorators": [],
            },
            {
                "type": "function",
                "function_name": "helper",
                "file_path": "module.py",
                "start_line": 7,
                "end_line": 10,
                "content": "def helper():\n    pass",
                "decorators": [],
            },
        ]

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create custom entry point via configuration
            analyzer = DeadCodeAnalyzer(custom_entry_points=["custom_entry"])
            report = analyzer.analyze(tmppath, chunks)

            # Both should be reachable via custom entry point
            assert report.reachable_count == 2
            assert report.unreachable_count == 0

    def test_filtering_by_min_confidence(self):
        """Test that findings are filtered by minimum confidence."""
        chunks = [
            {
                "type": "function",
                "function_name": "_private",  # HIGH confidence
                "file_path": "module.py",
                "start_line": 1,
                "end_line": 5,
                "content": "def _private(): pass",
                "decorators": [],
            },
            {
                "type": "function",
                "function_name": "public",  # MEDIUM confidence
                "file_path": "module.py",
                "start_line": 7,
                "end_line": 10,
                "content": "def public(): pass",
                "decorators": [],
            },
            {
                "type": "function",
                "function_name": "decorated",  # LOW confidence
                "file_path": "module.py",
                "start_line": 12,
                "end_line": 15,
                "content": "def decorated(): pass",
                "decorators": ["@app.route"],
            },
        ]

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Only HIGH confidence
            analyzer = DeadCodeAnalyzer(min_confidence=Confidence.HIGH)
            report = analyzer.analyze(tmppath, chunks)
            assert len(report.findings) == 1  # Only _private

            # MEDIUM and above
            analyzer = DeadCodeAnalyzer(min_confidence=Confidence.MEDIUM)
            report = analyzer.analyze(tmppath, chunks)
            assert len(report.findings) == 2  # _private and public

            # All findings
            analyzer = DeadCodeAnalyzer(min_confidence=Confidence.LOW)
            report = analyzer.analyze(tmppath, chunks)
            assert len(report.findings) == 3  # All three
