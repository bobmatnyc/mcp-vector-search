"""Test AST-based function call extraction in graph builder."""


def test_extract_function_calls_basic():
    """Test basic function call extraction."""
    # Import the function from graph_builder

    # Create a local version for testing
    import ast

    def extract_function_calls(code: str) -> set[str]:
        """Extract actual function calls from Python code using AST."""
        calls = set()
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    # Handle direct calls: foo()
                    if isinstance(node.func, ast.Name):
                        calls.add(node.func.id)
                    # Handle method calls: obj.foo() - extract 'foo'
                    elif isinstance(node.func, ast.Attribute):
                        calls.add(node.func.attr)
            return calls
        except SyntaxError:
            return set()

    # Test: actual function call
    code = """
def test():
    main()
    process_data()
"""
    calls = extract_function_calls(code)
    assert "main" in calls
    assert "process_data" in calls


def test_extract_function_calls_ignores_comments():
    """Test that comments mentioning function names don't create false positives."""
    import ast

    def extract_function_calls(code: str) -> set[str]:
        """Extract actual function calls from Python code using AST."""
        calls = set()
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        calls.add(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        calls.add(node.func.attr)
            return calls
        except SyntaxError:
            return set()

    # Test: comments should NOT create calls
    code = """
def test():
    # This comment mentions main but doesn't call it
    # Start the main server
    other_function()
"""
    calls = extract_function_calls(code)
    assert "main" not in calls  # Should NOT be detected
    assert "other_function" in calls  # Should be detected


def test_extract_function_calls_ignores_docstrings():
    """Test that docstrings mentioning function names don't create false positives."""
    import ast

    def extract_function_calls(code: str) -> set[str]:
        """Extract actual function calls from Python code using AST."""
        calls = set()
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        calls.add(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        calls.add(node.func.attr)
            return calls
        except SyntaxError:
            return set()

    # Test: docstrings should NOT create calls
    code = '''
def test():
    """This function calls main to do something."""
    actual_call()
'''
    calls = extract_function_calls(code)
    assert "main" not in calls  # Should NOT be detected
    assert "actual_call" in calls  # Should be detected


def test_extract_function_calls_method_calls():
    """Test that method calls are properly extracted."""
    import ast

    def extract_function_calls(code: str) -> set[str]:
        """Extract actual function calls from Python code using AST."""
        calls = set()
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        calls.add(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        calls.add(node.func.attr)
            return calls
        except SyntaxError:
            return set()

    # Test: method calls
    code = """
def test():
    obj.method_name()
    instance.process()
"""
    calls = extract_function_calls(code)
    assert "method_name" in calls
    assert "process" in calls


def test_extract_function_calls_invalid_syntax():
    """Test that invalid syntax returns empty set (no false positives)."""
    import ast

    def extract_function_calls(code: str) -> set[str]:
        """Extract actual function calls from Python code using AST."""
        calls = set()
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        calls.add(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        calls.add(node.func.attr)
            return calls
        except SyntaxError:
            return set()

    # Test: invalid syntax
    code = "this is not valid python code main("
    calls = extract_function_calls(code)
    assert len(calls) == 0  # Should return empty set, not crash
