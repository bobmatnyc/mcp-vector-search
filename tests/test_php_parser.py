#!/usr/bin/env python3
"""Test the PHP parser."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from mcp_vector_search.parsers.php import PHPParser


@pytest.mark.asyncio
async def test_php_parser():
    """Test PHP parser functionality."""
    print("🔍 Testing PHP parser...")

    # Create test PHP file
    php_content = """<?php

namespace App\\Services;

use App\\Models\\User;
use Illuminate\\Support\\Facades\\DB;

/**
 * User service for handling user operations
 * This class provides business logic for user management
 */
class UserService
{
    /**
     * @var User
     */
    private $user;

    /**
     * Initialize the user service
     *
     * @param User $user The user model instance
     */
    public function __construct(User $user)
    {
        $this->user = $user;
    }

    /**
     * Get user by ID
     *
     * @param int $userId User ID to fetch
     * @return User|null User instance or null if not found
     */
    public function getUserById(int $userId): ?User
    {
        return User::find($userId);
    }

    /**
     * Create a new user
     *
     * @param array $data User data
     * @return User Created user instance
     */
    public function createUser(array $data): User
    {
        return User::create([
            'name' => $data['name'],
            'email' => $data['email'],
            'password' => bcrypt($data['password'])
        ]);
    }

    /**
     * Update user information
     *
     * @param int $userId User ID
     * @param array $data Updated data
     * @return bool True if successful
     */
    public function updateUser(int $userId, array $data): bool
    {
        $user = $this->getUserById($userId);
        if (!$user) {
            return false;
        }

        return $user->update($data);
    }

    /**
     * Delete a user
     *
     * @param int $userId User ID to delete
     * @return bool True if successful
     */
    public function deleteUser(int $userId): bool
    {
        $user = $this->getUserById($userId);
        if (!$user) {
            return false;
        }

        return $user->delete();
    }

    /**
     * Get all active users
     *
     * @return array Array of active users
     */
    public static function getActiveUsers(): array
    {
        return User::where('status', 'active')->get()->toArray();
    }

    /**
     * Calculate user statistics
     *
     * @return array Statistics data
     */
    private function calculateStats(): array
    {
        return [
            'total' => User::count(),
            'active' => User::where('status', 'active')->count()
        ];
    }
}

/**
 * Authentication service interface
 * Defines contract for authentication implementations
 */
interface AuthServiceInterface
{
    /**
     * Authenticate user with credentials
     *
     * @param string $email User email
     * @param string $password User password
     * @return bool True if authenticated
     */
    public function authenticate(string $email, string $password): bool;

    /**
     * Logout the current user
     *
     * @return void
     */
    public function logout(): void;
}

/**
 * Logging trait for adding logging capabilities
 */
trait LoggableTrait
{
    /**
     * Log an info message
     *
     * @param string $message Message to log
     * @return void
     */
    protected function logInfo(string $message): void
    {
        error_log("[INFO] {$message}");
    }

    /**
     * Log an error message
     *
     * @param string $message Error message
     * @return void
     */
    protected function logError(string $message): void
    {
        error_log("[ERROR] {$message}");
    }
}

/**
 * Helper function to format user name
 *
 * @param string $firstName First name
 * @param string $lastName Last name
 * @return string Formatted full name
 */
function formatUserName(string $firstName, string $lastName): string
{
    return trim("{$firstName} {$lastName}");
}

/**
 * Helper function to validate email
 *
 * @param string $email Email address
 * @return bool True if valid email
 */
function isValidEmail(string $email): bool
{
    return filter_var($email, FILTER_VALIDATE_EMAIL) !== false;
}
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".php", delete=False) as f:
        f.write(php_content)
        test_file = Path(f.name)

    print(f"📁 Created test file: {test_file}")

    # Test PHP parser
    php_parser = PHPParser()
    chunks = await php_parser.parse_file(test_file)

    print(f"📊 PHP parser extracted {len(chunks)} chunks:")

    # Analyze chunks
    class_chunks = []
    interface_chunks = []
    trait_chunks = []
    method_chunks = []
    function_chunks = []

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
        if chunk.chunk_type == "class":
            class_chunks.append(chunk)
        if chunk.chunk_type == "interface":
            interface_chunks.append(chunk)
        if chunk.chunk_type == "trait":
            trait_chunks.append(chunk)
        if chunk.chunk_type == "method":
            method_chunks.append(chunk)
        if chunk.chunk_type == "function":
            function_chunks.append(chunk)

    # Verify key features
    print("\n" + "=" * 80)
    print("🎯 Feature Verification:")
    print("=" * 80)

    print(f"\n✅ Class chunks found: {len(class_chunks)}")
    assert len(class_chunks) >= 1, "Should find at least 1 class chunk"

    print(f"✅ Interface chunks found: {len(interface_chunks)}")
    assert len(interface_chunks) >= 1, "Should find at least 1 interface chunk"

    print(f"✅ Trait chunks found: {len(trait_chunks)}")
    assert len(trait_chunks) >= 1, "Should find at least 1 trait chunk"

    print(f"✅ Method chunks found: {len(method_chunks)}")
    assert len(method_chunks) >= 5, "Should find at least 5 method chunks"

    print(f"✅ Function chunks found: {len(function_chunks)}")
    assert len(function_chunks) >= 2, "Should find at least 2 function chunks"

    # Verify PHPDoc extraction
    chunks_with_docs = [c for c in chunks if c.docstring]
    print(f"✅ Chunks with PHPDoc: {len(chunks_with_docs)}/{len(chunks)}")
    assert len(chunks_with_docs) >= 5, "Should extract PHPDoc from multiple chunks"

    # Verify namespace handling (Note: Tree-sitter vs fallback may differ)
    # Tree-sitter implementation extracts namespace separately from classes
    # Fallback implementation applies namespace to class names
    namespaced_chunks = [c for c in chunks if c.class_name and "\\" in c.class_name]
    imports_chunks = [c for c in chunks if c.chunk_type == "imports"]
    print(f"✅ Namespaced chunks found: {len(namespaced_chunks)}")
    print(f"✅ Import chunks found: {len(imports_chunks)}")
    # Either approach is valid - namespace in class name OR separate import chunk
    assert len(namespaced_chunks) >= 1 or len(imports_chunks) >= 1, (
        "Should handle namespaces"
    )

    # Verify public/private/protected methods
    public_methods = [
        c for c in chunks if c.chunk_type == "method" and "public" in c.content
    ]
    print(f"✅ Public methods found: {len(public_methods)}")

    private_methods = [
        c for c in chunks if c.chunk_type == "method" and "private" in c.content
    ]
    print(f"✅ Private methods found: {len(private_methods)}")

    protected_methods = [
        c for c in chunks if c.chunk_type == "method" and "protected" in c.content
    ]
    print(f"✅ Protected methods found: {len(protected_methods)}")

    # Verify static methods
    static_methods = [
        c for c in chunks if c.chunk_type == "method" and "static" in c.content
    ]
    print(f"✅ Static methods found: {len(static_methods)}")
    assert len(static_methods) >= 1, "Should find static methods"

    # Verify supported extensions
    assert ".php" in php_parser.get_supported_extensions()
    assert ".phtml" in php_parser.get_supported_extensions()
    print(f"✅ Supported extensions: {php_parser.get_supported_extensions()}")

    # Clean up
    test_file.unlink()
    print("\n✅ PHP parser test completed successfully!")

    return True


@pytest.mark.asyncio
async def test_php_laravel_patterns():
    """Test PHP Laravel-specific patterns."""
    print("\n🔍 Testing PHP Laravel patterns...")

    # Create test file with Laravel-style patterns
    laravel_content = """<?php

namespace App\\Http\\Controllers;

use App\\Models\\Post;
use Illuminate\\Http\\Request;
use Illuminate\\Http\\JsonResponse;
use Illuminate\\Support\\Facades\\Auth;

/**
 * Post controller for managing blog posts
 */
class PostController extends Controller
{
    /**
     * Display a listing of posts
     *
     * @return \\Illuminate\\View\\View
     */
    public function index()
    {
        $posts = Post::with('user')->latest()->paginate(15);
        return view('posts.index', compact('posts'));
    }

    /**
     * Show the form for creating a new post
     *
     * @return \\Illuminate\\View\\View
     */
    public function create()
    {
        return view('posts.create');
    }

    /**
     * Store a newly created post
     *
     * @param Request $request HTTP request
     * @return \\Illuminate\\Http\\RedirectResponse
     */
    public function store(Request $request)
    {
        $validated = $request->validate([
            'title' => 'required|max:255',
            'content' => 'required',
        ]);

        $post = Auth::user()->posts()->create($validated);

        return redirect()->route('posts.show', $post)
            ->with('success', 'Post created successfully.');
    }

    /**
     * Display the specified post
     *
     * @param Post $post Post model instance
     * @return \\Illuminate\\View\\View
     */
    public function show(Post $post)
    {
        return view('posts.show', compact('post'));
    }

    /**
     * Update the specified post
     *
     * @param Request $request HTTP request
     * @param Post $post Post model instance
     * @return \\Illuminate\\Http\\RedirectResponse
     */
    public function update(Request $request, Post $post)
    {
        $this->authorize('update', $post);

        $validated = $request->validate([
            'title' => 'required|max:255',
            'content' => 'required',
        ]);

        $post->update($validated);

        return redirect()->route('posts.show', $post)
            ->with('success', 'Post updated successfully.');
    }

    /**
     * Remove the specified post
     *
     * @param Post $post Post model instance
     * @return \\Illuminate\\Http\\RedirectResponse
     */
    public function destroy(Post $post)
    {
        $this->authorize('delete', $post);
        $post->delete();

        return redirect()->route('posts.index')
            ->with('success', 'Post deleted successfully.');
    }
}

/**
 * Post model representing blog posts
 */
class Post extends Model
{
    /**
     * The attributes that are mass assignable
     *
     * @var array
     */
    protected $fillable = ['title', 'content', 'user_id'];

    /**
     * The attributes that should be cast
     *
     * @var array
     */
    protected $casts = [
        'published_at' => 'datetime',
    ];

    /**
     * Get the user that owns the post
     *
     * @return \\Illuminate\\Database\\Eloquent\\Relations\\BelongsTo
     */
    public function user()
    {
        return $this->belongsTo(User::class);
    }

    /**
     * Get the comments for the post
     *
     * @return \\Illuminate\\Database\\Eloquent\\Relations\\HasMany
     */
    public function comments()
    {
        return $this->hasMany(Comment::class);
    }

    /**
     * Scope a query to only include published posts
     *
     * @param \\Illuminate\\Database\\Eloquent\\Builder $query
     * @return \\Illuminate\\Database\\Eloquent\\Builder
     */
    public function scopePublished($query)
    {
        return $query->whereNotNull('published_at');
    }
}
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".php", delete=False) as f:
        f.write(laravel_content)
        test_file = Path(f.name)

    print(f"📁 Created Laravel test file: {test_file}")

    # Parse the file
    php_parser = PHPParser()
    chunks = await php_parser.parse_file(test_file)

    print(f"📊 Extracted {len(chunks)} chunks from Laravel patterns:")

    # Verify controller detection
    controller_classes = [
        c for c in chunks if c.class_name and "Controller" in c.class_name
    ]
    print(f"✅ Controller classes found: {len(controller_classes)}")

    # Verify model detection
    model_classes = [c for c in chunks if c.class_name and "Model" in str(c.content)]
    print(f"✅ Model classes found: {len(model_classes)}")

    # Verify CRUD methods
    crud_methods = ["index", "create", "store", "show", "update", "destroy"]
    found_crud = [
        c
        for c in chunks
        if c.function_name and any(crud in c.function_name for crud in crud_methods)
    ]
    print(f"✅ CRUD methods found: {len(found_crud)}")

    # Verify relationship methods
    relationship_methods = [
        c
        for c in chunks
        if c.function_name
        and any(rel in c.content for rel in ["belongsTo", "hasMany", "hasOne"])
    ]
    print(f"✅ Relationship methods found: {len(relationship_methods)}")

    # Clean up
    test_file.unlink()
    print("\n✅ Laravel patterns test completed successfully!")

    return True


@pytest.mark.asyncio
async def main():
    """Run all PHP parser tests."""
    try:
        await test_php_parser()
        await test_php_laravel_patterns()
        print("\n🎉 All PHP parser tests completed successfully!")
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
