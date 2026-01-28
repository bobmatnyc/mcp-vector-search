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
