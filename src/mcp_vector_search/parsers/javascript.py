"""JavaScript/TypeScript parser for MCP Vector Search."""

import re
from pathlib import Path

from loguru import logger

from ..core.models import CodeChunk
from .base import BaseParser


class JavaScriptParser(BaseParser):
    """JavaScript parser with tree-sitter AST support and fallback regex parsing."""

    def __init__(self, language: str = "javascript") -> None:
        """Initialize JavaScript parser with lazy grammar loading."""
        super().__init__(language)
        self._parser = None
        self._language = None
        self._use_tree_sitter = False
        self._initialized = False

    def _ensure_parser_initialized(self) -> None:
        """Ensure tree-sitter parser is initialized (lazy loading)."""
        if not self._initialized:
            self._initialize_parser()
            self._initialized = True

    def _initialize_parser(self) -> None:
        """Initialize Tree-sitter parser for JavaScript."""
        try:
            from tree_sitter_language_pack import get_language, get_parser

            self._language = get_language("javascript")
            self._parser = get_parser("javascript")
            self._use_tree_sitter = True
            return
        except Exception:
            self._use_tree_sitter = False

    async def parse_file(self, file_path: Path) -> list[CodeChunk]:
        """Parse a JavaScript/TypeScript file and extract code chunks."""
        try:
            with open(file_path, encoding="utf-8", errors="replace") as f:
                content = f.read()
            return await self.parse_content(content, file_path)
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return []

    async def parse_content(self, content: str, file_path: Path) -> list[CodeChunk]:
        """Parse JavaScript/TypeScript content and extract code chunks."""
        if not content.strip():
            return []

        # Lazy load parser on first use
        self._ensure_parser_initialized()

        if self._use_tree_sitter:
            try:
                tree = self._parser.parse(content.encode("utf-8"))
                return self._extract_chunks_from_tree(tree, content, file_path)
            except Exception as e:
                logger.warning(f"Tree-sitter parsing failed for {file_path}: {e}")
                return await self._regex_parse(content, file_path)
        else:
            return await self._regex_parse(content, file_path)

    def parse_file_sync(self, file_path: Path) -> list[CodeChunk]:
        """Parse file synchronously (optimized for multiprocessing workers)."""
        try:
            with open(file_path, encoding="utf-8", errors="replace") as f:
                content = f.read()
            return self._parse_content_sync(content, file_path)
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return []

    def _parse_content_sync(self, content: str, file_path: Path) -> list[CodeChunk]:
        """Parse content synchronously without async overhead."""
        if not content.strip():
            return []

        # Lazy load parser on first use
        self._ensure_parser_initialized()

        if self._use_tree_sitter:
            try:
                tree = self._parser.parse(content.encode("utf-8"))
                return self._extract_chunks_from_tree(tree, content, file_path)
            except Exception as e:
                logger.warning(f"Tree-sitter parsing failed for {file_path}: {e}")
                # Fallback to regex parse (need to run async in sync context)
                import asyncio

                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(
                        self._regex_parse(content, file_path)
                    )
                finally:
                    loop.close()
        else:
            # Use regex parse fallback
            import asyncio

            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self._regex_parse(content, file_path))
            finally:
                loop.close()

    def _extract_chunks_from_tree(
        self, tree, content: str, file_path: Path
    ) -> list[CodeChunk]:
        """Extract code chunks from JavaScript AST."""
        chunks = []
        lines = self._split_into_lines(content)

        # Collect top-level imports from the AST before walking nodes
        file_imports = self._extract_imports_from_tree(tree.root_node)

        def visit_node(node, current_class=None):
            """Recursively visit AST nodes."""
            node_type = node.type

            # Check if this node type should be extracted
            extracted = False

            if node_type == "function_declaration":
                chunks.extend(
                    self._extract_function(
                        node, lines, file_path, current_class, file_imports
                    )
                )
                extracted = True
            elif node_type == "arrow_function":
                chunks.extend(
                    self._extract_arrow_function(
                        node, lines, file_path, current_class, file_imports
                    )
                )
                extracted = True
            elif node_type == "class_declaration":
                class_chunks = self._extract_class(node, lines, file_path, file_imports)
                chunks.extend(class_chunks)

                # Visit class methods
                class_name = self._get_node_name(node)
                for child in node.children:
                    visit_node(child, class_name)
                extracted = True
            elif node_type == "method_definition":
                chunks.extend(
                    self._extract_method(
                        node, lines, file_path, current_class, file_imports
                    )
                )
                extracted = True
            elif node_type == "lexical_declaration":
                # const/let declarations might be arrow functions
                extracted_chunks = self._extract_variable_function(
                    node, lines, file_path, current_class, file_imports
                )
                if extracted_chunks:
                    chunks.extend(extracted_chunks)
                    extracted = True

            # Only recurse into children if we didn't extract this node
            # This prevents double-extraction of arrow functions in variable declarations
            if not extracted and hasattr(node, "children"):
                for child in node.children:
                    visit_node(child, current_class)

        visit_node(tree.root_node)

        # If no specific chunks found, create a single chunk for the whole file
        if not chunks:
            chunks.append(
                self._create_chunk(
                    content=content,
                    file_path=file_path,
                    start_line=1,
                    end_line=len(lines),
                    chunk_type="module",
                    imports=file_imports,
                )
            )

        return chunks

    def _extract_function(
        self,
        node,
        lines: list[str],
        file_path: Path,
        class_name: str | None = None,
        file_imports: list[dict] | None = None,
    ) -> list[CodeChunk]:
        """Extract function declaration from AST."""
        function_name = self._get_node_name(node)
        if not function_name:
            return []

        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        content = node.text.decode()
        docstring = self._extract_jsdoc_from_node(node, lines)

        # Calculate complexity
        complexity = self._calculate_complexity(node, "javascript")

        # Extract parameters
        parameters = self._extract_js_parameters(node)

        # Extract function calls made within this function body
        calls = self._extract_calls_from_node(node)

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="function",
            function_name=function_name,
            class_name=class_name,
            docstring=docstring,
            complexity_score=complexity,
            parameters=parameters,
            chunk_depth=2 if class_name else 1,
            imports=file_imports or [],
            calls=calls,
        )
        return [chunk]

    def _extract_arrow_function(
        self,
        node,
        lines: list[str],
        file_path: Path,
        class_name: str | None = None,
        file_imports: list[dict] | None = None,
    ) -> list[CodeChunk]:
        """Extract arrow function from AST."""
        # Arrow functions often don't have explicit names, try to get from parent
        parent = getattr(node, "parent", None)
        function_name = None

        if parent and parent.type == "variable_declarator":
            function_name = self._get_node_name(parent)

        if not function_name:
            return []

        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        content = node.text.decode()
        docstring = self._extract_jsdoc_from_node(node, lines)

        # Calculate complexity
        complexity = self._calculate_complexity(node, "javascript")

        # Extract parameters
        parameters = self._extract_js_parameters(node)

        # Extract function calls made within this arrow function body
        calls = self._extract_calls_from_node(node)

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="function",
            function_name=function_name,
            class_name=class_name,
            docstring=docstring,
            complexity_score=complexity,
            parameters=parameters,
            chunk_depth=2 if class_name else 1,
            imports=file_imports or [],
            calls=calls,
        )
        return [chunk]

    def _extract_variable_function(
        self,
        node,
        lines: list[str],
        file_path: Path,
        class_name: str | None = None,
        file_imports: list[dict] | None = None,
    ) -> list[CodeChunk]:
        """Extract function from variable declaration (const func = ...)."""
        chunks = []

        for child in node.children:
            if child.type == "variable_declarator":
                # Check if it's a function assignment
                for subchild in child.children:
                    if subchild.type in ("arrow_function", "function"):
                        func_name = self._get_node_name(child)
                        if func_name:
                            start_line = child.start_point[0] + 1
                            end_line = child.end_point[0] + 1

                            content = child.text.decode()
                            docstring = self._extract_jsdoc_from_node(child, lines)

                            # Calculate complexity
                            complexity = self._calculate_complexity(
                                subchild, "javascript"
                            )

                            # Extract parameters
                            parameters = self._extract_js_parameters(subchild)

                            # Extract function calls within this function body
                            calls = self._extract_calls_from_node(subchild)

                            chunk = self._create_chunk(
                                content=content,
                                file_path=file_path,
                                start_line=start_line,
                                end_line=end_line,
                                chunk_type="function",
                                function_name=func_name,
                                class_name=class_name,
                                docstring=docstring,
                                complexity_score=complexity,
                                parameters=parameters,
                                chunk_depth=2 if class_name else 1,
                                imports=file_imports or [],
                                calls=calls,
                            )
                            chunks.append(chunk)

        return chunks

    def _extract_class(
        self,
        node,
        lines: list[str],
        file_path: Path,
        file_imports: list[dict] | None = None,
    ) -> list[CodeChunk]:
        """Extract class declaration from AST."""
        class_name = self._get_node_name(node)
        if not class_name:
            return []

        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        content = node.text.decode()
        docstring = self._extract_jsdoc_from_node(node, lines)

        # Calculate complexity
        complexity = self._calculate_complexity(node, "javascript")

        # Extract base classes and implemented interfaces
        inherits_from = self._extract_class_heritage(node)

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="class",
            class_name=class_name,
            docstring=docstring,
            complexity_score=complexity,
            chunk_depth=1,
            imports=file_imports or [],
            inherits_from=inherits_from,
        )
        return [chunk]

    def _extract_method(
        self,
        node,
        lines: list[str],
        file_path: Path,
        class_name: str | None = None,
        file_imports: list[dict] | None = None,
    ) -> list[CodeChunk]:
        """Extract method definition from class."""
        method_name = self._get_node_name(node)
        if not method_name:
            return []

        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        content = node.text.decode()
        docstring = self._extract_jsdoc_from_node(node, lines)

        # Calculate complexity
        complexity = self._calculate_complexity(node, "javascript")

        # Extract parameters
        parameters = self._extract_js_parameters(node)

        # Check for decorators (TypeScript)
        decorators = self._extract_decorators_from_node(node)

        # Extract function calls made within this method body
        calls = self._extract_calls_from_node(node)

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="method",
            function_name=method_name,
            class_name=class_name,
            docstring=docstring,
            complexity_score=complexity,
            parameters=parameters,
            decorators=decorators,
            chunk_depth=2,
            imports=file_imports or [],
            calls=calls,
        )
        return [chunk]

    def _get_node_name(self, node) -> str | None:
        """Extract name from a named node."""
        for child in node.children:
            if child.type in ("identifier", "property_identifier", "type_identifier"):
                return child.text.decode("utf-8")
        return None

    def _get_node_text(self, node) -> str:
        """Get text content of a node."""
        if hasattr(node, "text"):
            return node.text.decode("utf-8")
        return ""

    def _extract_js_parameters(self, node) -> list[dict]:
        """Extract function parameters from JavaScript/TypeScript AST."""
        parameters = []

        for child in node.children:
            if child.type == "formal_parameters":
                for param_node in child.children:
                    if param_node.type in (
                        "identifier",
                        "required_parameter",
                        "optional_parameter",
                        "rest_parameter",
                    ):
                        param_info = {"name": None, "type": None, "default": None}

                        # Extract parameter details
                        if param_node.type == "identifier":
                            param_info["name"] = self._get_node_text(param_node)
                        else:
                            # TypeScript typed parameters
                            for subchild in param_node.children:
                                if subchild.type == "identifier":
                                    param_info["name"] = self._get_node_text(subchild)
                                elif subchild.type == "type_annotation":
                                    param_info["type"] = self._get_node_text(subchild)
                                elif (
                                    "default" in subchild.type
                                    or subchild.type == "number"
                                ):
                                    param_info["default"] = self._get_node_text(
                                        subchild
                                    )

                        if param_info["name"] and param_info["name"] not in (
                            "(",
                            ")",
                            ",",
                            "...",
                        ):
                            # Clean up rest parameters
                            if param_info["name"].startswith("..."):
                                param_info["name"] = param_info["name"][3:]
                                param_info["rest"] = True
                            parameters.append(param_info)

        return parameters

    def _extract_decorators_from_node(self, node) -> list[str]:
        """Extract decorators from TypeScript node."""
        decorators = []

        for child in node.children:
            if child.type == "decorator":
                decorators.append(self._get_node_text(child))

        return decorators

    def _extract_imports_from_tree(self, root_node) -> list[dict]:
        """Extract all import statements from the top-level AST.

        Collects both ES module import statements and CommonJS require calls
        that appear at the module top level.

        Args:
            root_node: Root node of the parsed AST

        Returns:
            List of import metadata dicts with 'statement' and 'source' keys
        """
        imports: list[dict] = []

        for child in root_node.children:
            # ES module: import X from 'Y'  /  import { X } from 'Y'
            if child.type == "import_statement":
                statement = child.text.decode("utf-8").strip()
                source = self._extract_import_source(child)
                imports.append({"statement": statement, "source": source})

            # CommonJS: const x = require('y')  /  var x = require('y')
            elif child.type in ("lexical_declaration", "variable_declaration"):
                for decl_child in child.children:
                    if decl_child.type == "variable_declarator":
                        for sub in decl_child.children:
                            if sub.type == "call_expression":
                                func_node = next(
                                    (
                                        c
                                        for c in sub.children
                                        if c.type == "identifier"
                                        and c.text.decode("utf-8") == "require"
                                    ),
                                    None,
                                )
                                if func_node is not None:
                                    statement = child.text.decode("utf-8").strip()
                                    source = self._extract_require_source(sub)
                                    imports.append(
                                        {"statement": statement, "source": source}
                                    )

            # export * from 'Y'  /  export { X } from 'Y'
            elif child.type == "export_statement":
                # Only record re-exports that have a source module
                source = self._extract_import_source(child)
                if source:
                    statement = child.text.decode("utf-8").strip()
                    imports.append({"statement": statement, "source": source})

        return imports

    def _extract_import_source(self, node) -> str:
        """Extract the source string from an import/export statement node.

        Args:
            node: import_statement or export_statement AST node

        Returns:
            Source module path string, or empty string if not found
        """
        for child in node.children:
            if child.type == "string":
                # Strip surrounding quotes
                return child.text.decode("utf-8").strip("\"'`")
        return ""

    def _extract_require_source(self, call_node) -> str:
        """Extract the source path from a require() call expression node.

        Args:
            call_node: call_expression AST node

        Returns:
            Required module path string, or empty string if not found
        """
        for child in call_node.children:
            if child.type == "arguments":
                for arg in child.children:
                    if arg.type == "string":
                        return arg.text.decode("utf-8").strip("\"'`")
        return ""

    def _extract_calls_from_node(self, node) -> list[str]:
        """Walk an AST node and collect all function/method call names.

        Traverses the full subtree collecting call_expression nodes.  For
        each call the callee name is extracted:
        - Simple call:  foo()         -> 'foo'
        - Method call:  obj.bar()     -> 'obj.bar'
        - Chained:      a.b.c()       -> 'a.b.c'

        Args:
            node: Root AST node to search within

        Returns:
            Deduplicated list of called function/method name strings
        """
        calls: list[str] = []
        seen: set[str] = set()

        def walk(n) -> None:
            if n.type == "call_expression":
                # First child of call_expression is the function being called
                if n.children:
                    callee = n.children[0]
                    name = self._get_callee_name(callee)
                    if name and name not in seen:
                        seen.add(name)
                        calls.append(name)
                # Still walk into arguments for nested calls
                for child in n.children:
                    walk(child)
            elif hasattr(n, "children"):
                for child in n.children:
                    walk(child)

        walk(node)
        return calls

    def _get_callee_name(self, node) -> str:
        """Extract a human-readable callee name from a call_expression callee node.

        Handles:
        - identifier:          foo
        - member_expression:   obj.method  (recursively flattened)

        Args:
            node: Callee AST node

        Returns:
            Dotted name string, or empty string if it cannot be resolved
        """
        if node.type == "identifier":
            return node.text.decode("utf-8")

        if node.type == "member_expression":
            # member_expression children: [object, ".", property]
            parts: list[str] = []
            for child in node.children:
                if child.type in ("identifier", "property_identifier"):
                    parts.append(child.text.decode("utf-8"))
                elif child.type == "member_expression":
                    parts.append(self._get_callee_name(child))
            return ".".join(p for p in parts if p)

        return ""

    def _extract_class_heritage(self, node) -> list[str]:
        """Extract base classes and implemented interfaces from a class node.

        Handles grammar differences between JavaScript and TypeScript:
        - JavaScript:   class_heritage -> (extends keyword) + identifier
        - TypeScript:   class_heritage -> extends_clause -> identifier
                        class_heritage -> implements_clause -> type_identifier

        Args:
            node: class_declaration AST node

        Returns:
            List of inherited/implemented class and interface name strings
        """
        inherits_from: list[str] = []

        for child in node.children:
            if child.type == "class_heritage":
                for heritage_child in child.children:
                    # TypeScript: explicit extends_clause wrapper
                    if heritage_child.type == "extends_clause":
                        for ext_child in heritage_child.children:
                            if ext_child.type in ("identifier", "type_identifier"):
                                inherits_from.append(ext_child.text.decode("utf-8"))
                            elif ext_child.type == "member_expression":
                                name = self._get_callee_name(ext_child)
                                if name:
                                    inherits_from.append(name)

                    # TypeScript: implements_clause
                    elif heritage_child.type == "implements_clause":
                        for impl_child in heritage_child.children:
                            if impl_child.type in ("identifier", "type_identifier"):
                                inherits_from.append(impl_child.text.decode("utf-8"))
                            elif impl_child.type == "generic_type":
                                # e.g. IFoo<Bar> - use only the outer name
                                for gc in impl_child.children:
                                    if gc.type in ("identifier", "type_identifier"):
                                        inherits_from.append(gc.text.decode("utf-8"))
                                        break

                    # JavaScript: class_heritage has no intermediate wrapper;
                    # identifiers appear directly as siblings of the 'extends' keyword
                    elif heritage_child.type in ("identifier", "type_identifier"):
                        inherits_from.append(heritage_child.text.decode("utf-8"))
                    elif heritage_child.type == "member_expression":
                        name = self._get_callee_name(heritage_child)
                        if name:
                            inherits_from.append(name)

        return inherits_from

    def _extract_jsdoc_from_node(self, node, lines: list[str]) -> str | None:
        """Extract JSDoc comment from before a node."""
        start_line = node.start_point[0]
        return self._extract_jsdoc(lines, start_line + 1)

    async def _regex_parse(self, content: str, file_path: Path) -> list[CodeChunk]:
        """Parse JavaScript/TypeScript using regex patterns."""
        chunks = []
        lines = self._split_into_lines(content)

        # JavaScript/TypeScript patterns
        function_patterns = [
            re.compile(r"^\s*function\s+(\w+)\s*\(", re.MULTILINE),  # function name()
            re.compile(
                r"^\s*const\s+(\w+)\s*=\s*\([^)]*\)\s*=>\s*{", re.MULTILINE
            ),  # const name = () => {
            re.compile(
                r"^\s*const\s+(\w+)\s*=\s*function\s*\(", re.MULTILINE
            ),  # const name = function(
            re.compile(
                r"^\s*(\w+)\s*:\s*function\s*\(", re.MULTILINE
            ),  # name: function(
            re.compile(r"^\s*(\w+)\s*\([^)]*\)\s*{", re.MULTILINE),  # name() { (method)
            re.compile(
                r"^\s*async\s+function\s+(\w+)\s*\(", re.MULTILINE
            ),  # async function name()
            re.compile(
                r"^\s*async\s+(\w+)\s*\([^)]*\)\s*{", re.MULTILINE
            ),  # async name() {
        ]

        class_patterns = [
            re.compile(r"^\s*class\s+(\w+)", re.MULTILINE),  # class Name
            re.compile(
                r"^\s*export\s+class\s+(\w+)", re.MULTILINE
            ),  # export class Name
            re.compile(
                r"^\s*export\s+default\s+class\s+(\w+)", re.MULTILINE
            ),  # export default class Name
        ]

        interface_patterns = [
            re.compile(
                r"^\s*interface\s+(\w+)", re.MULTILINE
            ),  # interface Name (TypeScript)
            re.compile(
                r"^\s*export\s+interface\s+(\w+)", re.MULTILINE
            ),  # export interface Name
        ]

        import_pattern = re.compile(r"^\s*(import|export).*", re.MULTILINE)

        # Extract imports
        imports = []
        for match in import_pattern.finditer(content):
            import_line = match.group(0).strip()
            imports.append(import_line)

        # Extract functions
        for pattern in function_patterns:
            for match in pattern.finditer(content):
                function_name = match.group(1)
                start_line = content[: match.start()].count("\n") + 1

                # Find end of function
                end_line = self._find_block_end(lines, start_line, "{", "}")

                func_content = self._get_line_range(lines, start_line, end_line)

                if func_content.strip():
                    # Extract JSDoc comment
                    jsdoc = self._extract_jsdoc(lines, start_line)

                    chunk = self._create_chunk(
                        content=func_content,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        chunk_type="function",
                        function_name=function_name,
                        docstring=jsdoc,
                    )
                    chunk.imports = imports
                    chunks.append(chunk)

        # Extract classes
        for pattern in class_patterns:
            for match in pattern.finditer(content):
                class_name = match.group(1)
                start_line = content[: match.start()].count("\n") + 1

                # Find end of class
                end_line = self._find_block_end(lines, start_line, "{", "}")

                class_content = self._get_line_range(lines, start_line, end_line)

                if class_content.strip():
                    # Extract JSDoc comment
                    jsdoc = self._extract_jsdoc(lines, start_line)

                    chunk = self._create_chunk(
                        content=class_content,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        chunk_type="class",
                        class_name=class_name,
                        docstring=jsdoc,
                    )
                    chunk.imports = imports
                    chunks.append(chunk)

        # Extract interfaces (TypeScript)
        if self.language == "typescript":
            for pattern in interface_patterns:
                for match in pattern.finditer(content):
                    interface_name = match.group(1)
                    start_line = content[: match.start()].count("\n") + 1

                    # Find end of interface
                    end_line = self._find_block_end(lines, start_line, "{", "}")

                    interface_content = self._get_line_range(
                        lines, start_line, end_line
                    )

                    if interface_content.strip():
                        # Extract JSDoc comment
                        jsdoc = self._extract_jsdoc(lines, start_line)

                        chunk = self._create_chunk(
                            content=interface_content,
                            file_path=file_path,
                            start_line=start_line,
                            end_line=end_line,
                            chunk_type="interface",
                            class_name=interface_name,  # Use class_name field for interface
                            docstring=jsdoc,
                        )
                        chunk.imports = imports
                        chunks.append(chunk)

        # If no specific chunks found, create a single chunk for the whole file
        if not chunks:
            chunks.append(
                self._create_chunk(
                    content=content,
                    file_path=file_path,
                    start_line=1,
                    end_line=len(lines),
                    chunk_type="module",
                )
            )

        return chunks

    def _find_block_end(
        self, lines: list[str], start_line: int, open_char: str, close_char: str
    ) -> int:
        """Find the end of a block by matching braces."""
        if start_line > len(lines):
            return len(lines)

        brace_count = 0
        found_opening = False

        for i in range(start_line - 1, len(lines)):
            line = lines[i]

            for char in line:
                if char == open_char:
                    brace_count += 1
                    found_opening = True
                elif char == close_char:
                    brace_count -= 1

                    if found_opening and brace_count == 0:
                        return i + 1  # Return 1-based line number

        return len(lines)

    def _extract_jsdoc(self, lines: list[str], start_line: int) -> str | None:
        """Extract JSDoc comment before a function/class."""
        if start_line <= 1:
            return None

        # Look backwards for JSDoc comment
        for i in range(start_line - 2, max(-1, start_line - 10), -1):
            line = lines[i].strip()

            if line.endswith("*/"):
                # Found end of JSDoc, collect the comment
                jsdoc_lines = []
                for j in range(i, -1, -1):
                    comment_line = lines[j].strip()
                    jsdoc_lines.insert(0, comment_line)

                    if comment_line.startswith("/**"):
                        # Found start of JSDoc
                        # Clean up the comment
                        cleaned_lines = []
                        for line in jsdoc_lines:
                            # Remove /** */ and * prefixes
                            cleaned = (
                                line.replace("/**", "")
                                .replace("*/", "")
                                .replace("*", "")
                                .strip()
                            )
                            if cleaned:
                                cleaned_lines.append(cleaned)

                        return " ".join(cleaned_lines) if cleaned_lines else None

            # If we hit non-comment code, stop looking
            elif line and not line.startswith("//") and not line.startswith("*"):
                break

        return None

    def get_supported_extensions(self) -> list[str]:
        """Get supported file extensions."""
        if self.language == "typescript":
            return [".ts", ".tsx"]
        else:
            return [".js", ".jsx", ".mjs"]


class TypeScriptParser(JavaScriptParser):
    """TypeScript parser extending JavaScript parser."""

    def __init__(self) -> None:
        """Initialize TypeScript parser."""
        super().__init__("typescript")

    def _initialize_parser(self) -> None:
        """Initialize Tree-sitter parser for TypeScript."""
        try:
            from tree_sitter_language_pack import get_language, get_parser

            self._language = get_language("typescript")
            self._parser = get_parser("typescript")
            self._use_tree_sitter = True
            return
        except Exception:
            self._use_tree_sitter = False
