"""Metadata extraction for Python code elements."""


class MetadataExtractor:
    """Extracts metadata like decorators, parameters, and return types from Python nodes."""

    @staticmethod
    def extract_decorators(node, lines: list[str]) -> list[str]:
        """Extract decorator names from function/class node.

        Args:
            node: Tree-sitter AST node
            lines: Source code lines (unused but kept for API consistency)

        Returns:
            List of decorator strings (including @ symbol)
        """
        decorators = []
        for child in node.children:
            if child.type == "decorator":
                # Get decorator text (includes @ symbol)
                dec_text = MetadataExtractor._get_node_text(child).strip()
                decorators.append(dec_text)
        return decorators

    @staticmethod
    def extract_parameters(node) -> list[dict]:
        """Extract function parameters with type annotations.

        Args:
            node: Tree-sitter function definition node

        Returns:
            List of parameter dictionaries with name, type, and default values
        """
        parameters = []
        for child in node.children:
            if child.type == "parameters":
                for param_node in child.children:
                    if param_node.type in (
                        "identifier",
                        "typed_parameter",
                        "default_parameter",
                    ):
                        param_info = {"name": None, "type": None, "default": None}

                        # Extract parameter name
                        if param_node.type == "identifier":
                            param_info["name"] = MetadataExtractor._get_node_text(
                                param_node
                            )
                        else:
                            # For typed or default parameters, find the identifier
                            for subchild in param_node.children:
                                if subchild.type == "identifier":
                                    param_info["name"] = (
                                        MetadataExtractor._get_node_text(subchild)
                                    )
                                elif subchild.type == "type":
                                    param_info["type"] = (
                                        MetadataExtractor._get_node_text(subchild)
                                    )
                                elif "default" in subchild.type:
                                    param_info["default"] = (
                                        MetadataExtractor._get_node_text(subchild)
                                    )

                        # Filter out special parameters and punctuation
                        if param_info["name"] and param_info["name"] not in (
                            "self",
                            "cls",
                            "(",
                            ")",
                            ",",
                        ):
                            parameters.append(param_info)
        return parameters

    @staticmethod
    def extract_return_type(node) -> str | None:
        """Extract return type annotation from function.

        Args:
            node: Tree-sitter function definition node

        Returns:
            Return type string or None
        """
        for child in node.children:
            if child.type == "type":
                return MetadataExtractor._get_node_text(child)
        return None

    @staticmethod
    def get_node_name(node) -> str | None:
        """Extract name from a named node (function, class, etc.).

        Args:
            node: Tree-sitter named node

        Returns:
            Node name or None
        """
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
        return None

    @staticmethod
    def _get_node_text(node) -> str:
        """Get text content of a node.

        Args:
            node: Tree-sitter node

        Returns:
            Node text content
        """
        if hasattr(node, "text"):
            return node.text.decode("utf-8")
        return ""

    @staticmethod
    def extract_function_calls(node, source_code: bytes) -> list[str]:
        """Extract function/method calls from AST node.

        Traverses the node tree to find all call_expression nodes and extracts
        the function names being called. Handles simple calls (print), method
        calls (self.save), and chained calls (obj.method).

        Args:
            node: Tree-sitter AST node to extract calls from
            source_code: Original source code as bytes (for text extraction)

        Returns:
            List of called function names (e.g., ['print', 'self.save', 'db.query'])
        """
        calls = []

        def _extract_call_name(call_node) -> str | None:
            """Extract function name from a call_expression node."""
            try:
                # Find the function being called (first child that's not '(' or ')')
                for child in call_node.children:
                    if child.type == "identifier":
                        # Simple function call: print()
                        return child.text.decode("utf-8")
                    elif child.type == "attribute":
                        # Method call: obj.method()
                        return child.text.decode("utf-8")
                    elif child.type in ("identifier", "attribute"):
                        # Fallback: get text
                        return child.text.decode("utf-8")
                return None
            except Exception:
                return None

        def _walk_tree(n):
            """Recursively walk tree to find call expressions."""
            try:
                if n.type == "call":
                    call_name = _extract_call_name(n)
                    if call_name and call_name not in calls:
                        calls.append(call_name)

                # Recurse to children
                for child in n.children:
                    _walk_tree(child)
            except Exception:
                # Silently ignore errors during traversal
                pass

        try:
            _walk_tree(node)
        except Exception:
            # Return empty list on error
            pass

        return calls

    @staticmethod
    def extract_imports(node, source_code: bytes) -> list[dict]:
        """Extract import statements from module-level AST.

        Finds import_statement and import_from_statement nodes and extracts
        module names, imported names, and aliases.

        Args:
            node: Tree-sitter AST node (typically module root)
            source_code: Original source code as bytes

        Returns:
            List of import dictionaries like:
            [
                {"module": "os", "names": ["path", "environ"], "alias": None},
                {"module": "typing", "names": ["List", "Dict"], "alias": None},
                {"module": "numpy", "names": ["*"], "alias": "np"}
            ]
        """
        imports = []

        def _extract_import_statement(import_node) -> list[dict] | None:
            """Extract data from 'import module' or 'import module as alias'.

            Can return multiple imports if comma-separated (import x, y, z).
            """
            try:
                results = []

                for child in import_node.children:
                    if child.type == "dotted_name" or child.type == "identifier":
                        # Simple import: import os
                        module_name = child.text.decode("utf-8")
                        if module_name not in ("import", ","):
                            results.append(
                                {
                                    "module": module_name,
                                    "names": ["*"],
                                    "alias": None,
                                }
                            )
                    elif child.type == "aliased_import":
                        # Handle 'import x as y'
                        module_name = None
                        alias = None
                        for subchild in child.children:
                            if subchild.type in ("dotted_name", "identifier"):
                                if module_name is None:
                                    module_name = subchild.text.decode("utf-8")
                                else:
                                    alias = subchild.text.decode("utf-8")
                        if module_name:
                            results.append(
                                {
                                    "module": module_name,
                                    "names": ["*"],
                                    "alias": alias,
                                }
                            )

                return results if results else None
            except Exception:
                return None

        def _extract_import_from_statement(import_node) -> dict | None:
            """Extract data from 'from module import names'."""
            try:
                module_name = None
                imported_names = []

                # First pass: find module name (before 'import' keyword)
                found_from = False
                for child in import_node.children:
                    if child.type == "from" or child.text == b"from":
                        found_from = True
                        continue

                    if found_from and child.type in ("dotted_name", "identifier"):
                        module_name = child.text.decode("utf-8")
                        break

                    if child.type == "import" or child.text == b"import":
                        break

                # Second pass: collect imported names (after 'import' keyword)
                collecting_names = False
                for child in import_node.children:
                    if child.type == "import" or child.text == b"import":
                        collecting_names = True
                        continue

                    if not collecting_names:
                        continue

                    if child.type == "wildcard_import" or child.text == b"*":
                        imported_names.append("*")
                    elif child.type == "dotted_name":
                        # Extract name from dotted_name node
                        imported_names.append(child.text.decode("utf-8"))
                    elif child.type == "identifier":
                        name = child.text.decode("utf-8")
                        if name not in (",", "as"):
                            imported_names.append(name)
                    elif child.type == "aliased_import":
                        # Handle 'from x import y as z' - extract first identifier/dotted_name
                        for subchild in child.children:
                            if subchild.type in ("identifier", "dotted_name"):
                                imported_names.append(subchild.text.decode("utf-8"))
                                break

                if module_name and imported_names:
                    return {
                        "module": module_name,
                        "names": imported_names,
                        "alias": None,
                    }
                return None
            except Exception:
                return None

        def _walk_tree(n):
            """Walk tree to find import statements."""
            try:
                if n.type == "import_statement":
                    import_data = _extract_import_statement(n)
                    if import_data:
                        # Can be list or single dict
                        if isinstance(import_data, list):
                            imports.extend(import_data)
                        else:
                            imports.append(import_data)
                elif n.type == "import_from_statement":
                    import_data = _extract_import_from_statement(n)
                    if import_data:
                        imports.append(import_data)

                # Only traverse module-level children (don't go into functions/classes)
                if n.type in ("module", "expression_statement"):
                    for child in n.children:
                        _walk_tree(child)
            except Exception:
                pass

        try:
            _walk_tree(node)
        except Exception:
            pass

        return imports

    @staticmethod
    def extract_class_bases(node, source_code: bytes) -> list[str]:
        """Extract base classes from class definition.

        Looks for argument_list in class_definition node to extract parent classes.

        Args:
            node: Tree-sitter class_definition node
            source_code: Original source code as bytes

        Returns:
            List of base class names like: ['BaseModel', 'ABC', 'Generic[T]']
        """
        bases = []

        try:
            # Find argument_list node (contains base classes)
            for child in node.children:
                if child.type == "argument_list":
                    # Extract each base class from argument list
                    for arg_child in child.children:
                        if arg_child.type in ("identifier", "attribute", "subscript"):
                            base_name = arg_child.text.decode("utf-8")
                            if base_name and base_name not in ("(", ")", ","):
                                bases.append(base_name)
        except Exception:
            # Return empty list on error
            pass

        return bases
