#!/usr/bin/env python3
"""Test the JavaScript parser."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from mcp_vector_search.parsers.javascript import JavaScriptParser, TypeScriptParser


@pytest.mark.asyncio
async def test_javascript_parser():
    """Test JavaScript parser functionality."""
    print("🔍 Testing JavaScript parser...")

    # Create test JavaScript file
    js_content = """
import React from 'react';
import { useState } from 'react';

/**
 * A simple counter component
 * @param {Object} props - Component props
 * @returns {JSX.Element} Counter component
 */
function Counter(props) {
    const [count, setCount] = useState(0);

    const increment = () => {
        setCount(count + 1);
    };

    return (
        <div>
            <p>Count: {count}</p>
            <button onClick={increment}>Increment</button>
        </div>
    );
}

/**
 * Utility class for data processing
 */
class DataProcessor {
    constructor() {
        this.data = [];
    }

    /**
     * Add an item to the data array
     * @param {any} item - Item to add
     */
    addItem(item) {
        this.data.push(item);
    }

    /**
     * Process all data items
     * @returns {Array} Processed data
     */
    processAll() {
        return this.data.map(item => item.toString().toUpperCase());
    }
}

// Arrow function example
const calculateArea = (radius) => {
    return Math.PI * radius * radius;
};

// Async function example
async function fetchData(url) {
    try {
        const response = await fetch(url);
        return await response.json();
    } catch (error) {
        console.error('Error fetching data:', error);
        return null;
    }
}

export default Counter;
export { DataProcessor, calculateArea, fetchData };
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
        f.write(js_content)
        test_file = Path(f.name)

    print(f"📁 Created test file: {test_file}")

    # Test JavaScript parser
    js_parser = JavaScriptParser()
    chunks = await js_parser.parse_file(test_file)

    print(f"📊 JavaScript parser extracted {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n📄 Chunk {i}:")
        print(f"  Type: {chunk.chunk_type}")
        print(f"  Lines: {chunk.start_line}-{chunk.end_line}")
        print(f"  Function: {chunk.function_name}")
        print(f"  Class: {chunk.class_name}")
        print(f"  Docstring: {chunk.docstring}")
        print(f"  Content preview: {chunk.content[:100]}...")

    # Clean up
    test_file.unlink()
    print("\n✅ JavaScript parser test completed!")


@pytest.mark.asyncio
async def test_typescript_parser():
    """Test TypeScript parser functionality."""
    print("\n🔍 Testing TypeScript parser...")

    # Create test TypeScript file
    ts_content = """
interface User {
    id: number;
    name: string;
    email: string;
}

/**
 * User service for managing user data
 */
class UserService {
    private users: User[] = [];

    /**
     * Add a new user
     * @param user - User to add
     */
    addUser(user: User): void {
        this.users.push(user);
    }

    /**
     * Find user by ID
     * @param id - User ID
     * @returns User or undefined
     */
    findById(id: number): User | undefined {
        return this.users.find(user => user.id === id);
    }
}

/**
 * Generic API response interface
 */
interface ApiResponse<T> {
    data: T;
    status: number;
    message: string;
}

/**
 * Fetch user data from API
 * @param id - User ID
 * @returns Promise with user data
 */
async function fetchUser(id: number): Promise<ApiResponse<User>> {
    const response = await fetch(`/api/users/${id}`);
    return response.json();
}

// Type alias
type UserCallback = (user: User) => void;

// Generic function
function processUsers<T extends User>(users: T[], callback: UserCallback): void {
    users.forEach(callback);
}

export { UserService, fetchUser, processUsers };
export type { User, ApiResponse, UserCallback };
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".ts", delete=False) as f:
        f.write(ts_content)
        test_file = Path(f.name)

    print(f"📁 Created test file: {test_file}")

    # Test TypeScript parser
    ts_parser = TypeScriptParser()
    chunks = await ts_parser.parse_file(test_file)

    print(f"📊 TypeScript parser extracted {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n📄 Chunk {i}:")
        print(f"  Type: {chunk.chunk_type}")
        print(f"  Lines: {chunk.start_line}-{chunk.end_line}")
        print(f"  Function: {chunk.function_name}")
        print(f"  Class: {chunk.class_name}")
        print(f"  Docstring: {chunk.docstring}")
        print(f"  Content preview: {chunk.content[:100]}...")

    # Clean up
    test_file.unlink()
    print("\n✅ TypeScript parser test completed!")


@pytest.mark.asyncio
async def main():
    """Run all parser tests."""
    await test_javascript_parser()
    await test_typescript_parser()
    print("\n🎉 All parser tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
