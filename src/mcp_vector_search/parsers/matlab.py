"""MATLAB parser for MCP Vector Search."""

import re
from pathlib import Path

from loguru import logger

from ..core.models import CodeChunk
from .base import BaseParser


class MatlabParser(BaseParser):
    """MATLAB parser with tree-sitter AST support and fallback regex parsing.

    Uses the ``acristoffers/tree-sitter-matlab`` grammar (bundled in
    tree-sitter-language-pack) to extract functions, classes (``classdef``),
    methods, and script-level code from ``.m`` files.

    Note on the ``.m`` extension: ``.m`` is shared with Objective-C. This
    project has no Objective-C parser, so ``.m`` is mapped to MATLAB. If
    Objective-C support is ever added, the registry will need disambiguation
    (e.g. by content sniffing).
    """

    def __init__(self) -> None:
        """Initialize MATLAB parser."""
        super().__init__("matlab")
        self._parser = None
        self._language = None
        self._use_tree_sitter = False
        self._initialized = False

    def _initialize_parser(self) -> None:
        """Initialize Tree-sitter parser for MATLAB."""
        try:
            from tree_sitter_language_pack import get_language, get_parser

            self._language = get_language("matlab")
            self._parser = get_parser("matlab")
            # Verify the parser is actually callable — version mismatches between
            # tree_sitter_language_pack and the installed tree-sitter C extension
            # can produce a 'builtins.Parser' object that lacks .parse()
            if not callable(getattr(self._parser, "parse", None)):
                raise RuntimeError(
                    f"tree_sitter_language_pack returned an incompatible parser "
                    f"(type={type(self._parser).__module__}.{type(self._parser).__name__}). "
                    f"Run: pip install --upgrade tree-sitter tree-sitter-language-pack"
                )
            self._use_tree_sitter = True
            return
        except Exception:
            self._use_tree_sitter = False

    def _ensure_parser_initialized(self) -> None:
        """Ensure tree-sitter parser is initialized (lazy loading)."""
        if not self._initialized:
            self._initialize_parser()
            self._initialized = True

    async def parse_file(self, file_path: Path) -> list[CodeChunk]:
        """Parse a MATLAB file and extract code chunks."""
        try:
            file_bytes = file_path.read_bytes()
            content = file_bytes.decode("utf-8", errors="replace")
            return await self.parse_content(content, file_path)
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return []

    async def parse_content(self, content: str, file_path: Path) -> list[CodeChunk]:
        """Parse MATLAB content and extract code chunks."""
        if not content.strip():
            return []

        # Lazy load parser on first use
        self._ensure_parser_initialized()

        if self._use_tree_sitter:
            try:
                content_bytes = content.encode("utf-8")
                tree = self._parser.parse(content_bytes)
                return self._extract_chunks_from_tree(tree, content, file_path)
            except Exception as e:
                logger.warning(f"Tree-sitter parsing failed for {file_path}: {e}")
                return await self._regex_parse(content, file_path)
        else:
            return await self._regex_parse(content, file_path)

    def _extract_chunks_from_tree(
        self, tree, content: str, file_path: Path
    ) -> list[CodeChunk]:
        """Extract code chunks from MATLAB AST."""
        chunks = []
        lines = self._split_into_lines(content)

        def visit_node(node, current_class=None):
            """Recursively visit AST nodes."""
            node_type = node.type

            if node_type == "function_definition":
                # A function (or a method when inside a classdef). We do not
                # recurse into the body, so nested functions are folded into
                # their parent chunk's content (kept simple and overlap-free).
                chunks.extend(
                    self._extract_function(node, lines, file_path, current_class)
                )
            elif node_type == "class_definition":
                class_name = self._get_node_name(node)
                chunks.extend(self._extract_class(node, lines, file_path, class_name))
                # Recurse so the methods block's function_definitions are
                # tagged with the enclosing class name.
                for child in node.children:
                    visit_node(child, class_name)
            else:
                for child in node.children:
                    visit_node(child, current_class)

        visit_node(tree.root_node)

        # If no specific chunks found (e.g. a plain script), create a single
        # chunk for the whole file.
        if not chunks:
            chunks.append(
                self._create_chunk(
                    content=content,
                    file_path=file_path,
                    start_line=1,
                    end_line=len(lines),
                    chunk_type="script",
                )
            )

        return chunks

    def _extract_function(
        self, node, lines: list[str], file_path: Path, current_class: str | None
    ) -> list[CodeChunk]:
        """Extract a function_definition node as a chunk."""
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        function_name = self._get_node_name(node)
        content = self._get_line_range(lines, start_line, end_line)

        docstring = self._extract_docstring(node)
        complexity = self._calculate_complexity(node, "matlab")
        parameters = self._extract_parameters(node)
        return_type = self._extract_outputs(node)
        calls = self._extract_calls(node)

        # Inside a classdef methods block this is a method; otherwise a
        # top-level (or local) function.
        chunk_type = "method" if current_class else "function"

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type=chunk_type,
            function_name=function_name,
            class_name=current_class,
            docstring=docstring,
            complexity_score=complexity,
            parameters=parameters,
            return_type=return_type,
            calls=calls,
            chunk_depth=2 if current_class else 1,
        )

        return [chunk]

    def _extract_class(
        self, node, lines: list[str], file_path: Path, class_name: str
    ) -> list[CodeChunk]:
        """Extract a class_definition (classdef) node as a chunk."""
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        content = self._get_line_range(lines, start_line, end_line)
        docstring = self._extract_docstring(node)
        inherits_from = self._extract_superclasses(node)

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="class",
            class_name=class_name,
            docstring=docstring,
            inherits_from=inherits_from,
            chunk_depth=1,
        )

        return [chunk]

    def _get_node_name(self, node) -> str:
        """Extract the name of a function or class definition.

        For both ``function_definition`` and ``class_definition`` the name is
        the direct ``identifier`` child (distinct from ``function_output`` and
        ``function_arguments`` nodes, which have their own types).
        """
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
        return "unknown"

    def _extract_parameters(self, node) -> list[dict]:
        """Extract parameter names from a function_definition node."""
        parameters: list[dict] = []
        for child in node.children:
            if child.type == "function_arguments":
                for arg in child.children:
                    if arg.type == "identifier":
                        parameters.append({"name": arg.text.decode("utf-8")})
                break
        return parameters

    def _extract_outputs(self, node) -> str | None:
        """Extract output variable names (MATLAB's analogue of a return type).

        Returns a string such as ``"hdr_loc"`` or ``"[a, b, c]"``, or None when
        the function has no outputs.
        """
        for child in node.children:
            if child.type == "function_output":
                names: list[str] = []
                for sub in child.children:
                    if sub.type == "identifier":
                        names.append(sub.text.decode("utf-8"))
                    elif sub.type == "multioutput_variable":
                        for ident in sub.children:
                            if ident.type == "identifier":
                                names.append(ident.text.decode("utf-8"))
                if not names:
                    return None
                if len(names) == 1:
                    return names[0]
                return "[" + ", ".join(names) + "]"
        return None

    def _extract_superclasses(self, node) -> list[str]:
        """Extract base class names from a class_definition's superclasses."""
        bases: list[str] = []
        for child in node.children:
            if child.type == "superclasses":
                for prop in child.children:
                    if prop.type == "property_name":
                        # property_name may be a dotted name
                        # (e.g. matlab.mixin.Copyable); use its full text.
                        bases.append(prop.text.decode("utf-8"))
        return bases

    def _extract_docstring(self, node) -> str | None:
        """Extract the leading help comment of a function/class definition.

        In MATLAB the help text is a comment immediately following the
        signature; the grammar attaches it as a direct ``comment`` child of the
        definition node (before the ``block``).
        """
        for child in node.children:
            if child.type == "block":
                # Past the signature; no leading comment was found.
                break
            if child.type == "comment":
                return self._clean_comment(child.text.decode("utf-8"))
        return None

    def _clean_comment(self, text: str) -> str:
        """Strip MATLAB comment markers (% and %{ %}) from comment text."""
        lines = text.splitlines()
        cleaned: list[str] = []
        for raw in lines:
            stripped = raw.strip()
            if stripped in ("%{", "%}"):
                continue
            # Remove a leading run of % characters and one optional space.
            stripped = re.sub(r"^%+\s?", "", stripped)
            cleaned.append(stripped)
        return "\n".join(cleaned).strip()

    def _extract_calls(self, node) -> list[str]:
        """Extract called names from a function/method subtree.

        Walks for ``function_call`` and ``command`` nodes and records the
        callee identifier. Note: MATLAB does not syntactically distinguish a
        function call ``f(x)`` from array indexing ``A(i)``, so this list is an
        over-approximation that may include indexed local variables.
        """
        calls: list[str] = []
        seen: set[str] = set()

        def add(name: str) -> None:
            if name and len(name) >= 2 and name not in seen:
                seen.add(name)
                calls.append(name)

        def walk(n) -> None:
            try:
                if n.type == "function_call":
                    first = n.children[0] if n.children else None
                    if first is not None:
                        if first.type == "identifier":
                            add(first.text.decode("utf-8"))
                        elif first.type == "field_expression":
                            # e.g. obj.method(...) -> use the last identifier
                            last_ident = None
                            for sub in first.children:
                                if sub.type == "identifier":
                                    last_ident = sub.text.decode("utf-8")
                            if last_ident:
                                add(last_ident)
                elif n.type == "command":
                    # Command syntax: `hold on`, `format long`, etc. First
                    # child names the command.
                    if n.children and n.children[0].type == "command_name":
                        add(n.children[0].text.decode("utf-8"))
                for child in n.children:
                    walk(child)
            except Exception:
                pass

        try:
            walk(node)
        except Exception:
            pass

        return calls

    def _calculate_complexity(self, node, language: str | None = None) -> float:
        """Calculate cyclomatic complexity for MATLAB code."""
        if not hasattr(node, "children"):
            return 1.0

        complexity = 1.0  # Base complexity

        decision_nodes = {
            "if_statement",
            "elseif_clause",
            "while_statement",
            "for_statement",
            "case_clause",
            "catch_clause",
            "boolean_operator",  # && and || (short-circuit and element-wise)
        }

        def count_decision_points(n):
            nonlocal complexity
            if hasattr(n, "type") and n.type in decision_nodes:
                complexity += 1
            if hasattr(n, "children"):
                for child in n.children:
                    count_decision_points(child)

        count_decision_points(node)
        return complexity

    async def _regex_parse(self, content: str, file_path: Path) -> list[CodeChunk]:
        """Fallback regex-based parsing when tree-sitter is unavailable.

        MATLAB blocks are ``end``-delimited, but ``end`` is heavily overloaded
        (it also closes if/for/while/switch/try and indexes arrays, e.g.
        ``x(end)``), so reliable brace-style matching is not possible. As an
        approximation we slice each definition from its header line to the line
        before the next top-level definition.
        """
        lines = self._split_into_lines(content)

        func_re = re.compile(
            r"^\s*function\b\s*(?:\[?[\w,\s]*\]?\s*=\s*)?(\w+)", re.IGNORECASE
        )
        class_re = re.compile(r"^\s*classdef\b\s*(?:\([^)]*\)\s*)?(\w+)", re.IGNORECASE)

        # Find every definition header line (1-based) with its kind/name.
        defs: list[tuple[int, str, str]] = []
        for i, line in enumerate(lines):
            m = class_re.match(line)
            if m:
                defs.append((i + 1, "class", m.group(1)))
                continue
            m = func_re.match(line)
            if m:
                defs.append((i + 1, "function", m.group(1)))

        chunks: list[CodeChunk] = []
        for idx, (start_line, kind, name) in enumerate(defs):
            end_line = (defs[idx + 1][0] - 1) if idx + 1 < len(defs) else len(lines)
            block = self._get_line_range(lines, start_line, end_line)
            if not block.strip():
                continue
            chunks.append(
                self._create_chunk(
                    content=block,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type=kind,
                    function_name=name if kind == "function" else None,
                    class_name=name if kind == "class" else None,
                )
            )

        # If no specific chunks found, create a single chunk for the whole file.
        if not chunks:
            chunks.append(
                self._create_chunk(
                    content=content,
                    file_path=file_path,
                    start_line=1,
                    end_line=len(lines),
                    chunk_type="script",
                )
            )

        return chunks

    def parse_file_sync(self, file_path: Path) -> list[CodeChunk]:
        """Parse file synchronously (optimized for multiprocessing workers)."""
        try:
            file_bytes = file_path.read_bytes()
            content = file_bytes.decode("utf-8", errors="replace")
            return self._parse_content_sync(content, file_path)
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return []

    def _parse_content_sync(self, content: str, file_path: Path) -> list[CodeChunk]:
        """Parse content synchronously without async overhead."""
        if not content.strip():
            return []

        self._ensure_parser_initialized()

        if self._use_tree_sitter:
            try:
                content_bytes = content.encode("utf-8")
                tree = self._parser.parse(content_bytes)
                return self._extract_chunks_from_tree(tree, content, file_path)
            except Exception as e:
                logger.warning(f"Tree-sitter parsing failed for {file_path}: {e}")
                return self._run_regex_sync(content, file_path)
        else:
            return self._run_regex_sync(content, file_path)

    def _run_regex_sync(self, content: str, file_path: Path) -> list[CodeChunk]:
        """Run the async regex fallback from a synchronous context."""
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._regex_parse(content, file_path))
        finally:
            loop.close()

    def get_supported_extensions(self) -> list[str]:
        """Get supported file extensions."""
        return [".m"]
