"""Language-specific profiles for multi-language PR/MR review support.

This module provides language detection and language-specific standards
(idioms, anti-patterns, security concerns) for comprehensive code review
across multiple programming languages.

Design Philosophy:
    - Detect languages from file extensions in PR patches
    - Provide language-specific idioms and conventions
    - Flag language-specific anti-patterns
    - Highlight language-specific security vulnerabilities
    - Support mixed-language PRs gracefully
    - Extensible to new languages

Supported Languages:
    Python, TypeScript, JavaScript, Java, C#, Ruby, Go, Rust, PHP, Swift, Kotlin, Scala
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LanguageProfile:
    """Language-specific standards and patterns for code review.

    Attributes:
        name: Human-readable language name (e.g., "Python", "TypeScript")
        extensions: File extensions for this language (e.g., [".py", ".pyw"])
        config_files: Tool config files to discover (e.g., ["pyproject.toml"])
        idioms: Common language-specific standards and best practices
        anti_patterns: Language-specific bad practices to flag
        security_patterns: Language-specific security concerns to check

    Example:
        >>> profile = LANGUAGE_PROFILES["python"]
        >>> print(profile.name)
        Python
        >>> print(profile.extensions)
        ['.py', '.pyw', '.pyi']
    """

    name: str
    extensions: list[str] = field(default_factory=list)
    config_files: list[str] = field(default_factory=list)
    idioms: list[str] = field(default_factory=list)
    anti_patterns: list[str] = field(default_factory=list)
    security_patterns: list[str] = field(default_factory=list)


# Language profiles with standards, idioms, and security patterns
LANGUAGE_PROFILES: dict[str, LanguageProfile] = {
    "python": LanguageProfile(
        name="Python",
        extensions=[".py", ".pyw", ".pyi"],
        config_files=["pyproject.toml", ".flake8", "setup.cfg", "mypy.ini", "ruff.toml"],
        idioms=[
            "Use type hints for all function signatures (PEP 484)",
            "Follow PEP 8 style guide (snake_case, 79-88 char lines)",
            "Use dataclasses or Pydantic for data models",
            "Prefer context managers (with) for resource management",
            "Use f-strings for string formatting (Python 3.6+)",
            "Use pathlib.Path instead of os.path",
            "Prefer list/dict/set comprehensions over map/filter",
            "Use is None instead of == None for None checks",
        ],
        anti_patterns=[
            "Mutable default arguments (def f(x=[]))",
            "Catching bare except: without specifying exception type",
            "Using == None instead of is None",
            "String concatenation in loops (use join() instead)",
            "Wildcard imports (from module import *)",
            "Not using __all__ in modules meant for import",
            "Using type() for type checks instead of isinstance()",
        ],
        security_patterns=[
            "SQL injection via string formatting in queries",
            "Command injection via subprocess/os.system with user input",
            "Unsafe deserialization (pickle.loads on untrusted data)",
            "Hardcoded secrets/credentials in code",
            "Path traversal in file operations",
            "eval() or exec() with user input",
        ],
    ),
    "typescript": LanguageProfile(
        name="TypeScript",
        extensions=[".ts", ".tsx", ".mts", ".cts"],
        config_files=["tsconfig.json", ".eslintrc.js", ".eslintrc.json", "prettier.config.js"],
        idioms=[
            "Use strict TypeScript (strict: true in tsconfig)",
            "Prefer interface over type for object shapes",
            "Use readonly for immutable properties",
            "Prefer optional chaining (?.) over null checks",
            "Use nullish coalescing (??) over || for null/undefined",
            "Use const assertions (as const) for literal types",
            "Avoid any — use unknown for unsafe values",
            "Prefer async/await over Promise.then() chains",
        ],
        anti_patterns=[
            "Using any type (disables type safety)",
            "Non-null assertions (!) without justification",
            "Implicit any in function parameters",
            "Mutating arrays/objects passed as parameters",
            "Mixing async/await and .then() chains",
            "Using var instead of const/let",
        ],
        security_patterns=[
            "XSS via innerHTML/dangerouslySetInnerHTML",
            "Prototype pollution vulnerabilities",
            "Insecure eval() usage",
            "Hardcoded API keys in client code",
            "CSRF in API calls without tokens",
            "Missing input sanitization for DOM manipulation",
        ],
    ),
    "javascript": LanguageProfile(
        name="JavaScript",
        extensions=[".js", ".jsx", ".mjs", ".cjs"],
        config_files=[".eslintrc.js", ".eslintrc.json", "prettier.config.js", ".prettierrc"],
        idioms=[
            "Use const/let instead of var",
            "Prefer arrow functions for callbacks",
            "Use destructuring for object/array access",
            "Use template literals instead of string concatenation",
            "Prefer async/await over Promise.then() chains",
            "Use optional chaining (?.) and nullish coalescing (??)",
            "Use strict equality (===) instead of ==",
        ],
        anti_patterns=[
            "Using var instead of const/let",
            "Using == instead of === for comparisons",
            "Modifying function arguments",
            "Not handling Promise rejections",
            "Callback hell (deeply nested callbacks)",
            "Global variable pollution",
        ],
        security_patterns=[
            "XSS via innerHTML or document.write",
            "eval() with user input",
            "Prototype pollution",
            "Insecure random number generation (Math.random for security)",
            "CSRF vulnerabilities",
            "Missing CORS validation",
        ],
    ),
    "java": LanguageProfile(
        name="Java",
        extensions=[".java"],
        config_files=["checkstyle.xml", "pmd.xml", "spotbugs.xml", "build.gradle", "pom.xml"],
        idioms=[
            "Follow Java naming conventions (camelCase methods, PascalCase classes)",
            "Use Optional<T> instead of null returns",
            "Prefer immutable objects (final fields)",
            "Use try-with-resources for AutoCloseable",
            "Override equals() and hashCode() together",
            "Use StringBuilder for string concatenation in loops",
            "Prefer interfaces over abstract classes for type definitions",
            "Use EnumSet instead of bit fields",
        ],
        anti_patterns=[
            "Returning null instead of Optional or empty collections",
            "Catching Exception or Throwable without re-throwing",
            "Using raw types (List instead of List<T>)",
            "String concatenation in loops with +",
            "Mutable public fields",
            "Not closing resources (streams, connections)",
            "Empty catch blocks",
        ],
        security_patterns=[
            "SQL injection in JDBC queries",
            "XXE injection in XML parsing",
            "Deserialization of untrusted data",
            "Path traversal in file operations",
            "Insecure random (Math.random() for security)",
            "Hardcoded credentials",
        ],
    ),
    "csharp": LanguageProfile(
        name="C#",
        extensions=[".cs", ".csx"],
        config_files=[".editorconfig", "stylecop.json", "*.csproj"],
        idioms=[
            "Follow Microsoft naming conventions (PascalCase for public members)",
            "Use async/await for async operations (not .Result/.Wait())",
            "Use LINQ for collection operations",
            "Prefer using declarations (C# 8+)",
            "Use record types for immutable data (C# 9+)",
            "Use nullable reference types (enable in .csproj)",
            "Use IDisposable pattern for unmanaged resources",
            "Prefer string interpolation over String.Format",
        ],
        anti_patterns=[
            "Blocking async code with .Result or .Wait()",
            "Catching general Exception without logging",
            "Not disposing IDisposable objects",
            "Using dynamic type unnecessarily",
            "String concatenation in loops without StringBuilder",
            "Empty catch blocks",
        ],
        security_patterns=[
            "SQL injection in ADO.NET or string-built queries",
            "XSS in ASP.NET Razor views",
            "Insecure deserialization (BinaryFormatter)",
            "Hardcoded connection strings",
            "Open redirect vulnerabilities",
            "Missing input validation in ASP.NET controllers",
        ],
    ),
    "ruby": LanguageProfile(
        name="Ruby",
        extensions=[".rb", ".rake", ".gemspec"],
        config_files=[".rubocop.yml", "Gemfile", ".rubocop_todo.yml"],
        idioms=[
            "Follow Ruby style guide (snake_case, 2-space indent)",
            "Use guard clauses instead of nested conditionals",
            "Prefer duck typing over type checking",
            "Use symbols for hash keys when possible",
            "Use blocks/procs/lambdas idiomatically",
            "Prefer map/select/reduce over explicit loops",
            "Use frozen string literals (# frozen_string_literal: true)",
            "Use safe navigation operator (&.)",
        ],
        anti_patterns=[
            "Rescue Exception instead of StandardError",
            "Using global variables ($var)",
            "Long method chains without intermediate variables",
            "Checking class with .class instead of is_a?",
            "String mutation in tight loops",
            "Not using trailing commas in multi-line collections",
        ],
        security_patterns=[
            "SQL injection via ActiveRecord string interpolation",
            "Mass assignment vulnerabilities",
            "Command injection via backticks or system()",
            "CSRF in Rails controllers",
            "Insecure deserialization (YAML.load on untrusted data)",
            "Missing strong parameters in Rails",
        ],
    ),
    "go": LanguageProfile(
        name="Go",
        extensions=[".go"],
        config_files=[".golangci.yml", "go.mod", "go.sum"],
        idioms=[
            "Return errors as last return value, check immediately",
            "Use goroutines and channels for concurrency",
            "Accept interfaces, return structs",
            "Use defer for cleanup (file.Close(), mutex.Unlock())",
            "Keep functions small and focused",
            "Use table-driven tests",
            "Name receiver variables consistently (1-2 letter abbreviations)",
            "Use context.Context for cancellation",
        ],
        anti_patterns=[
            "Ignoring error return values (_)",
            "Using panic for control flow",
            "Goroutine leaks (goroutines without cancellation)",
            "Copying mutex values",
            "Using init() for complex initialization",
            "Premature optimization without profiling",
        ],
        security_patterns=[
            "SQL injection via fmt.Sprintf in queries",
            "Command injection via exec.Command with user input",
            "Path traversal in file operations",
            "Race conditions in concurrent code",
            "Hardcoded credentials",
            "Unsafe use of crypto/rand vs math/rand",
        ],
    ),
    "rust": LanguageProfile(
        name="Rust",
        extensions=[".rs"],
        config_files=["Cargo.toml", "clippy.toml", ".clippy.toml", "rustfmt.toml"],
        idioms=[
            "Use Result<T,E> for recoverable errors, panic! for unrecoverable",
            "Prefer ownership and borrowing over clone()",
            "Use iterators instead of explicit loops",
            "Use ? operator for error propagation",
            "Use derive macros (Debug, Clone, etc.) appropriately",
            "Prefer &str over String for function parameters",
            "Use pattern matching exhaustively",
            "Follow Rust naming conventions (snake_case for functions)",
        ],
        anti_patterns=[
            "Excessive .clone() calls hiding ownership issues",
            "Using unwrap()/expect() in library code",
            "Unsafe blocks without clear justification",
            "Blocking in async context (std::thread::sleep in async)",
            "Ignoring Result with let _ = ",
            "Not using #[must_use] for critical return types",
        ],
        security_patterns=[
            "Unsafe code without clear justification and documentation",
            "Integer overflow in unsafe arithmetic",
            "Use-after-free patterns in unsafe code",
            "Unchecked indexing that could panic",
            "Missing bounds checking in unsafe code",
        ],
    ),
    "php": LanguageProfile(
        name="PHP",
        extensions=[".php", ".phtml"],
        config_files=[".php-cs-fixer.php", "phpcs.xml", "phpstan.neon", "composer.json"],
        idioms=[
            "Use strict types declaration (declare(strict_types=1))",
            "Follow PSR-12 coding standard",
            "Use type hints for parameters and return types",
            "Use composer autoloading (namespaces)",
            "Prefer PDO with prepared statements for DB",
            "Use named arguments for clarity (PHP 8+)",
            "Use null coalescing operator (??)",
        ],
        anti_patterns=[
            "Using mysql_* functions (deprecated)",
            "Mixing HTML and PHP logic",
            "Using == instead of === for comparisons",
            "Suppressing errors with @ operator",
            "Global variables and global keyword",
            "Not using prepared statements for SQL",
        ],
        security_patterns=[
            "SQL injection via string interpolation in queries",
            "XSS via unescaped output (echo $userInput)",
            "CSRF in forms without tokens",
            "Path traversal in include/require",
            "Remote code execution via eval()",
            "File upload vulnerabilities",
        ],
    ),
    "swift": LanguageProfile(
        name="Swift",
        extensions=[".swift"],
        config_files=[".swiftlint.yml", "Package.swift"],
        idioms=[
            "Use guard for early returns and unwrapping",
            "Prefer let over var for immutability",
            "Use optional chaining (?.) and nil coalescing (??)",
            "Follow Swift naming conventions (camelCase)",
            "Use extensions for protocol conformance",
            "Prefer value types (struct) over reference types (class)",
            "Use trailing closure syntax when appropriate",
        ],
        anti_patterns=[
            "Force unwrapping (!) without justification",
            "Using var when let would suffice",
            "Long functions without extracted methods",
            "Not handling optional values properly",
            "Using NSObject when Swift types suffice",
        ],
        security_patterns=[
            "SQL injection in CoreData predicates",
            "Insecure data storage (UserDefaults for sensitive data)",
            "Missing SSL pinning for network calls",
            "Insecure random number generation",
            "Hardcoded API keys",
        ],
    ),
    "kotlin": LanguageProfile(
        name="Kotlin",
        extensions=[".kt", ".kts"],
        config_files=["build.gradle.kts", "detekt.yml"],
        idioms=[
            "Use val over var for immutability",
            "Use data classes for DTOs",
            "Prefer nullable types over null checks",
            "Use when expression instead of if-else chains",
            "Use extension functions for utility methods",
            "Use scope functions (let, apply, run, with) appropriately",
            "Follow Kotlin naming conventions",
        ],
        anti_patterns=[
            "Using !! (not-null assertion) without justification",
            "Not using data classes when appropriate",
            "Long functions without extracted methods",
            "Ignoring nullable types",
            "Mixing Java and Kotlin idioms unnecessarily",
        ],
        security_patterns=[
            "SQL injection in Room queries",
            "Insecure data storage (SharedPreferences for sensitive data)",
            "Hardcoded credentials",
            "Missing input validation",
            "Insecure network communication",
        ],
    ),
    "scala": LanguageProfile(
        name="Scala",
        extensions=[".scala", ".sc"],
        config_files=["build.sbt", "scalafmt.conf", ".scalafmt.conf"],
        idioms=[
            "Prefer immutable collections",
            "Use case classes for domain models",
            "Use pattern matching exhaustively",
            "Prefer Option over null",
            "Use for-comprehensions for monadic operations",
            "Follow Scala naming conventions (camelCase)",
            "Use lazy val for expensive computations",
        ],
        anti_patterns=[
            "Using null instead of Option",
            "Not handling Option/Either properly",
            "Mutable collections in public APIs",
            "Long functions without decomposition",
            "Mixing styles (Java vs Scala idioms)",
        ],
        security_patterns=[
            "SQL injection in Slick queries",
            "Deserialization vulnerabilities",
            "Hardcoded credentials",
            "Missing input validation",
            "Insecure XML parsing",
        ],
    ),
}


def detect_languages(patches: list) -> dict[str, list[str]]:
    """Detect languages from PR patches based on file extensions.

    Args:
        patches: List of PRFilePatch objects

    Returns:
        Dictionary mapping language name to list of file paths in that language

    Example:
        >>> patches = [PRFilePatch(file_path="src/main.py", ...), PRFilePatch(file_path="src/utils.ts", ...)]
        >>> detect_languages(patches)
        {'python': ['src/main.py'], 'typescript': ['src/utils.ts']}
    """
    language_files: dict[str, list[str]] = {}

    for patch in patches:
        # Skip deleted files
        if hasattr(patch, "is_deleted") and patch.is_deleted:
            continue

        file_path = patch.file_path
        extension = Path(file_path).suffix.lower()

        # Find matching language profile
        for lang_key, profile in LANGUAGE_PROFILES.items():
            if extension in profile.extensions:
                if lang_key not in language_files:
                    language_files[lang_key] = []
                language_files[lang_key].append(file_path)
                break  # Only match first language

    return language_files


def get_profile(file_path: str) -> LanguageProfile | None:
    """Get language profile from file extension.

    Args:
        file_path: Path to file

    Returns:
        LanguageProfile if extension matches a known language, None otherwise

    Example:
        >>> profile = get_profile("src/auth.py")
        >>> print(profile.name)
        Python
    """
    extension = Path(file_path).suffix.lower()

    for profile in LANGUAGE_PROFILES.values():
        if extension in profile.extensions:
            return profile

    return None


def get_languages_in_pr(patches: list) -> list[LanguageProfile]:
    """Get unique language profiles present in a PR.

    Args:
        patches: List of PRFilePatch objects

    Returns:
        List of unique LanguageProfile objects present in the PR

    Example:
        >>> patches = [PRFilePatch(file_path="src/main.py", ...), PRFilePatch(file_path="src/utils.py", ...)]
        >>> profiles = get_languages_in_pr(patches)
        >>> [p.name for p in profiles]
        ['Python']
    """
    language_keys = set()

    for patch in patches:
        # Skip deleted files
        if hasattr(patch, "is_deleted") and patch.is_deleted:
            continue

        file_path = patch.file_path
        extension = Path(file_path).suffix.lower()

        # Find matching language profile
        for lang_key, profile in LANGUAGE_PROFILES.items():
            if extension in profile.extensions:
                language_keys.add(lang_key)
                break  # Only match first language

    # Return profiles in consistent order (alphabetical by language name)
    return sorted(
        [LANGUAGE_PROFILES[key] for key in language_keys], key=lambda p: p.name
    )
