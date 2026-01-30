"""Node extractors for different Python code elements."""

from pathlib import Path

from ...core.models import CodeChunk
from .class_skeleton_generator import ClassSkeletonGenerator
from .docstring_extractor import DocstringExtractor
from .metadata_extractor import MetadataExtractor


class NodeExtractorBase:
    """Base class for node extractors."""

    def __init__(self, base_parser):
        """Initialize with reference to base parser.

        Args:
            base_parser: Reference to the PythonParser instance for shared utilities
        """
        self.base_parser = base_parser


class FunctionExtractor(NodeExtractorBase):
    """Extracts function definitions as code chunks."""

    def extract(
        self, node, lines: list[str], file_path: Path, class_name: str | None = None
    ) -> list[CodeChunk]:
        """Extract function definition as a chunk.

        Args:
            node: Tree-sitter function definition node
            lines: Source code lines
            file_path: Path to source file
            class_name: Parent class name if this is a method

        Returns:
            List containing the function chunk
        """
        chunks = []

        function_name = MetadataExtractor.get_node_name(node)
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        # Get function content
        content = node.text.decode()

        # Extract docstring if present
        docstring = DocstringExtractor.extract_from_node(node, lines)

        # Calculate complexity
        complexity = self.base_parser._calculate_complexity(node, "python")

        # Extract decorators
        decorators = MetadataExtractor.extract_decorators(node, lines)

        # Extract parameters
        parameters = MetadataExtractor.extract_parameters(node)

        # Extract return type
        return_type = MetadataExtractor.extract_return_type(node)

        chunk = self.base_parser._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="function",
            function_name=function_name,
            class_name=class_name,
            docstring=docstring,
            complexity_score=complexity,
            decorators=decorators,
            parameters=parameters,
            return_type=return_type,
            chunk_depth=2 if class_name else 1,
        )
        chunks.append(chunk)

        return chunks


class ClassExtractor(NodeExtractorBase):
    """Extracts class definitions as code chunks."""

    def extract(self, node, lines: list[str], file_path: Path) -> list[CodeChunk]:
        """Extract class definition as a chunk (skeleton only, no method bodies).

        Args:
            node: Tree-sitter class definition node
            lines: Source code lines
            file_path: Path to source file

        Returns:
            List containing the class chunk
        """
        chunks = []

        class_name = MetadataExtractor.get_node_name(node)
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        # Get class skeleton (without method bodies)
        content = ClassSkeletonGenerator.generate_from_node(node, lines)

        # Extract docstring if present
        docstring = DocstringExtractor.extract_from_node(node, lines)

        # Calculate complexity (for the entire class)
        complexity = self.base_parser._calculate_complexity(node, "python")

        # Extract decorators
        decorators = MetadataExtractor.extract_decorators(node, lines)

        chunk = self.base_parser._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="class",
            class_name=class_name,
            docstring=docstring,
            complexity_score=complexity,
            decorators=decorators,
            chunk_depth=1,
        )
        chunks.append(chunk)

        return chunks


class ModuleExtractor(NodeExtractorBase):
    """Extracts module-level code like imports."""

    def extract(self, node, lines: list[str], file_path: Path) -> CodeChunk | None:
        """Extract module-level code (imports, constants, etc.).

        Args:
            node: Tree-sitter module node
            lines: Source code lines
            file_path: Path to source file

        Returns:
            Module chunk or None if no module-level code found
        """
        # Look for module-level statements (not inside functions/classes)
        module_lines = []

        for child in node.children:
            if child.type in ["import_statement", "import_from_statement"]:
                import_content = child.text.decode()
                module_lines.append(import_content.strip())

        if module_lines:
            content = "\n".join(module_lines)
            return self.base_parser._create_chunk(
                content=content,
                file_path=file_path,
                start_line=1,
                end_line=len(module_lines),
                chunk_type="imports",
            )

        return None
