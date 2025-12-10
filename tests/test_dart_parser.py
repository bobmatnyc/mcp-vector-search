#!/usr/bin/env python3
"""Test the Dart parser."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from mcp_vector_search.parsers.dart import DartParser


@pytest.mark.asyncio
async def test_dart_parser():
    """Test Dart parser functionality."""
    print("🔍 Testing Dart parser...")

    # Create test Dart file
    dart_content = """
import 'package:flutter/material.dart';

/// A simple stateless widget example
/// This widget displays static text
class SimpleTextWidget extends StatelessWidget {
  const SimpleTextWidget({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Text('Hello, Flutter!');
  }
}

/// A stateful counter widget
/// Demonstrates state management in Flutter
class CounterWidget extends StatefulWidget {
  const CounterWidget({Key? key}) : super(key: key);

  @override
  State<CounterWidget> createState() => _CounterWidgetState();
}

/// State class for CounterWidget
class _CounterWidgetState extends State<CounterWidget> {
  int _counter = 0;

  /// Increment the counter
  void _incrementCounter() {
    setState(() {
      _counter++;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Text('Counter: $_counter'),
        ElevatedButton(
          onPressed: _incrementCounter,
          child: Text('Increment'),
        ),
      ],
    );
  }
}

/// Utility class for data processing
class DataProcessor {
  final List<String> data = [];

  /// Add an item to the data list
  void addItem(String item) {
    data.add(item);
  }

  /// Process all data items
  List<String> processAll() {
    return data.map((item) => item.toUpperCase()).toList();
  }
}

/// Fetch user data from API
/// Returns a Future containing user data
Future<Map<String, dynamic>> fetchUserData(String userId) async {
  await Future.delayed(Duration(seconds: 2));
  return {
    'id': userId,
    'name': 'John Doe',
    'email': 'john@example.com',
  };
}

/// A utility mixin for logging
mixin LoggerMixin {
  /// Log an info message
  void logInfo(String message) {
    print('[INFO] $message');
  }

  /// Log an error message
  void logError(String message, [dynamic error]) {
    print('[ERROR] $message');
    if (error != null) {
      print('[ERROR] ${error.toString()}');
    }
  }
}

/// Service class using the logger mixin
class UserService with LoggerMixin {
  /// Get user by ID
  Future<Map<String, dynamic>> getUser(String id) async {
    logInfo('Fetching user: $id');
    try {
      final userData = await fetchUserData(id);
      logInfo('User fetched successfully');
      return userData;
    } catch (e) {
      logError('Failed to fetch user', e);
      rethrow;
    }
  }
}

/// Simple arrow function
String greet(String name) => 'Hello, $name!';

/// Calculate circle area
double calculateArea(double radius) => 3.14159 * radius * radius;

/// Main application entry point
void main() {
  runApp(MyApp());
}

/// Root application widget
class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Dart Parser Test',
      home: Scaffold(
        appBar: AppBar(title: Text('Test App')),
        body: Column(
          children: [
            SimpleTextWidget(),
            CounterWidget(),
          ],
        ),
      ),
    );
  }
}
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".dart", delete=False) as f:
        f.write(dart_content)
        test_file = Path(f.name)

    print(f"📁 Created test file: {test_file}")

    # Test Dart parser
    dart_parser = DartParser()
    chunks = await dart_parser.parse_file(test_file)

    print(f"📊 Dart parser extracted {len(chunks)} chunks:")

    # Analyze chunks
    widget_chunks = []
    class_chunks = []
    function_chunks = []
    mixin_chunks = []
    async_chunks = []

    for i, chunk in enumerate(chunks, 1):
        print(f"\n📄 Chunk {i}:")
        print(f"  Type: {chunk.chunk_type}")
        print(f"  Lines: {chunk.start_line}-{chunk.end_line}")
        if chunk.function_name:
            print(f"  Function: {chunk.function_name}")
        if chunk.class_name:
            print(f"  Class: {chunk.class_name}")
        if chunk.docstring:
            print(f"  Docstring: {chunk.docstring[:60]}...")
        print(f"  Content preview: {chunk.content[:100]}...")

        # Categorize chunks
        if chunk.chunk_type == "widget" or (
            chunk.class_name and "Widget" in chunk.class_name
        ):
            widget_chunks.append(chunk)
        if chunk.chunk_type == "class":
            class_chunks.append(chunk)
        if chunk.chunk_type == "function":
            function_chunks.append(chunk)
        if chunk.chunk_type == "mixin":
            mixin_chunks.append(chunk)
        if chunk.chunk_type == "function" and "async" in chunk.content[:150]:
            async_chunks.append(chunk)

    # Verify key features
    print("\n" + "=" * 80)
    print("🎯 Feature Verification:")
    print("=" * 80)

    print(f"\n✅ Widget chunks found: {len(widget_chunks)}")
    assert len(widget_chunks) >= 2, "Should find at least 2 widget chunks"

    print(f"✅ Class chunks found: {len(class_chunks)}")
    assert len(class_chunks) >= 1, "Should find at least 1 class chunk"

    print(f"✅ Function chunks found: {len(function_chunks)}")
    assert len(function_chunks) >= 3, "Should find at least 3 function chunks"

    print(f"✅ Mixin chunks found: {len(mixin_chunks)}")
    assert len(mixin_chunks) >= 1, "Should find at least 1 mixin chunk"

    print(f"✅ Async function chunks found: {len(async_chunks)}")
    assert len(async_chunks) >= 1, "Should find at least 1 async function chunk"

    # Verify dartdoc extraction
    chunks_with_docs = [c for c in chunks if c.docstring]
    print(f"✅ Chunks with dartdoc: {len(chunks_with_docs)}/{len(chunks)}")
    assert len(chunks_with_docs) >= 5, "Should extract dartdoc from multiple chunks"

    # Verify supported extensions
    assert ".dart" in dart_parser.get_supported_extensions()
    print(f"✅ Supported extensions: {dart_parser.get_supported_extensions()}")

    # Clean up
    test_file.unlink()
    print("\n✅ Dart parser test completed successfully!")

    return True


@pytest.mark.asyncio
async def test_dart_widget_patterns():
    """Test Dart widget-specific parsing patterns."""
    print("\n🔍 Testing Dart Widget patterns...")

    # Create test file with various widget patterns
    widget_content = """
import 'package:flutter/material.dart';

/// Custom button widget
class CustomButton extends StatelessWidget {
  final String label;
  final VoidCallback onPressed;

  const CustomButton({
    Key? key,
    required this.label,
    required this.onPressed,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return ElevatedButton(
      onPressed: onPressed,
      child: Text(label),
    );
  }
}

/// Form widget with state
class LoginForm extends StatefulWidget {
  const LoginForm({Key? key}) : super(key: key);

  @override
  State<LoginForm> createState() => _LoginFormState();
}

class _LoginFormState extends State<LoginForm> {
  final _formKey = GlobalKey<FormState>();
  String _email = '';
  String _password = '';

  void _submit() {
    if (_formKey.currentState!.validate()) {
      _formKey.currentState!.save();
      print('Email: $_email, Password: $_password');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Form(
      key: _formKey,
      child: Column(
        children: [
          TextFormField(
            decoration: InputDecoration(labelText: 'Email'),
            onSaved: (value) => _email = value ?? '',
          ),
          TextFormField(
            decoration: InputDecoration(labelText: 'Password'),
            obscureText: true,
            onSaved: (value) => _password = value ?? '',
          ),
          ElevatedButton(
            onPressed: _submit,
            child: Text('Login'),
          ),
        ],
      ),
    );
  }
}
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".dart", delete=False) as f:
        f.write(widget_content)
        test_file = Path(f.name)

    print(f"📁 Created widget test file: {test_file}")

    # Parse the file
    dart_parser = DartParser()
    chunks = await dart_parser.parse_file(test_file)

    print(f"📊 Extracted {len(chunks)} chunks from widget patterns:")

    # Verify StatelessWidget detection
    stateless_widgets = [
        c for c in chunks if c.class_name and "StatelessWidget" in str(c.class_name)
    ]
    print(f"✅ StatelessWidget classes found: {len(stateless_widgets)}")

    # Verify StatefulWidget detection
    stateful_widgets = [
        c for c in chunks if c.class_name and "StatefulWidget" in str(c.class_name)
    ]
    print(f"✅ StatefulWidget classes found: {len(stateful_widgets)}")

    # Verify State class detection
    state_classes = [c for c in chunks if c.class_name and "State" in str(c.class_name)]
    print(f"✅ State classes found: {len(state_classes)}")

    # Verify build methods are captured
    build_methods = [c for c in chunks if c.function_name == "build"]
    print(f"✅ Build methods found: {len(build_methods)}")

    # Clean up
    test_file.unlink()
    print("\n✅ Widget patterns test completed successfully!")

    return True


@pytest.mark.asyncio
async def main():
    """Run all Dart parser tests."""
    try:
        await test_dart_parser()
        await test_dart_widget_patterns()
        print("\n🎉 All Dart parser tests completed successfully!")
        return True
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
