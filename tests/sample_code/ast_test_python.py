#!/usr/bin/env python3
"""Sample Python file for testing AST enhancements.

This file tests:
1. Complexity score calculation (multiple if/while/for statements)
2. Decorator extraction (@property, @staticmethod, etc.)
3. Parameter extraction with type annotations
4. Return type extraction
5. Hierarchical chunk relationships (module → class → method)
"""

from dataclasses import dataclass


# Module-level function (depth 1, parent = module)
def simple_function(name: str) -> str:
    """Simple function with low complexity (score = 1)."""
    return f"Hello, {name}"


# Function with moderate complexity (score = 5)
def calculate_grade(score: int, bonus: int = 0) -> str:
    """Calculate letter grade with moderate complexity.

    Args:
        score: Base score (0-100)
        bonus: Bonus points to add

    Returns:
        Letter grade (A, B, C, D, F)
    """
    total = score + bonus

    if total >= 90:  # +1
        return "A"
    elif total >= 80:  # +1
        return "B"
    elif total >= 70:  # +1
        return "C"
    elif total >= 60:  # +1
        return "D"
    else:
        return "F"


# Function with high complexity (score = 10+)
def complex_validator(data: dict[str, any]) -> list[str] | None:
    """Validate data with high complexity.

    Tests multiple decision points: if, for, while, try/except.
    """
    errors = []

    if not data:  # +1
        return ["Data is empty"]

    if "name" not in data:  # +1
        errors.append("Missing name field")
    elif len(data["name"]) < 3:  # +1
        errors.append("Name too short")

    if "age" in data:  # +1
        try:  # +1
            age = int(data["age"])
            if age < 0 or age > 150:  # +1
                errors.append("Invalid age range")
        except ValueError:
            errors.append("Age must be a number")

    if "tags" in data:  # +1
        for tag in data["tags"]:  # +1
            if not isinstance(tag, str):  # +1
                errors.append(f"Invalid tag type: {tag}")

    return errors if errors else None  # +1 (conditional expression)


@dataclass
class User:
    """User class with various method types (depth 1, parent = module).

    Tests:
    - Class chunk extraction
    - Method chunks (depth 2, parent = User class)
    - Decorator extraction (@property, @staticmethod, etc.)
    - Parameter type annotations
    """

    name: str
    age: int
    email: str | None = None

    def __post_init__(self):
        """Validate user data after initialization."""
        if self.age < 0:
            raise ValueError("Age cannot be negative")

    @property
    def display_name(self) -> str:
        """Get formatted display name."""
        return self.name.title()

    @property
    def is_adult(self) -> bool:
        """Check if user is an adult (complexity = 2)."""
        if self.age >= 18:  # +1
            return True
        return False

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format (moderate complexity).

        Args:
            email: Email address to validate

        Returns:
            True if email is valid
        """
        if not email:  # +1
            return False

        if "@" not in email:  # +1
            return False

        parts = email.split("@")
        if len(parts) != 2:  # +1
            return False

        return True

    @classmethod
    def from_dict(cls, data: dict[str, any]) -> "User":
        """Create user from dictionary."""
        return cls(
            name=data.get("name", ""), age=data.get("age", 0), email=data.get("email")
        )

    def update_profile(
        self, name: str | None = None, age: int | None = None, email: str | None = None
    ) -> None:
        """Update user profile with optional fields.

        Tests parameter extraction with Optional types and defaults.
        """
        if name is not None:  # +1
            self.name = name
        if age is not None:  # +1
            self.age = age
        if email is not None:  # +1
            self.email = email


class AuthenticationManager:
    """Authentication manager with complex methods (depth 1).

    Tests hierarchical relationships with nested complexity.
    """

    def __init__(self, secret_key: str, timeout: int = 3600):
        """Initialize authentication manager.

        Args:
            secret_key: Secret key for token generation
            timeout: Token timeout in seconds (default: 1 hour)
        """
        self.secret_key = secret_key
        self.timeout = timeout
        self._cache: dict[str, any] = {}

    def authenticate(
        self, username: str, password: str, remember_me: bool = False
    ) -> str | None:
        """Authenticate user and return token (high complexity).

        Args:
            username: Username to authenticate
            password: User password
            remember_me: Whether to extend token lifetime

        Returns:
            Authentication token or None if failed
        """
        if not username or not password:  # +1
            return None

        # Check cache first
        if username in self._cache:  # +1
            cached = self._cache[username]
            if cached["password"] == password:  # +1
                if remember_me:  # +1
                    cached["timeout"] = self.timeout * 24  # 24 hours
                return cached["token"]

        # Validate credentials
        valid = False
        try:  # +1
            valid = self._validate_credentials(username, password)
        except Exception as e:
            print(f"Validation error: {e}")
            return None

        if not valid:  # +1
            return None

        # Generate token
        token = self._generate_token(username)

        # Cache result
        self._cache[username] = {
            "password": password,
            "token": token,
            "timeout": self.timeout * 24 if remember_me else self.timeout,  # +1
        }

        return token

    def _validate_credentials(self, username: str, password: str) -> bool:
        """Validate username and password (private method).

        Tests private method extraction.
        """
        if len(password) < 8:  # +1
            return False

        if not any(c.isupper() for c in password):  # +1
            return False

        if not any(c.isdigit() for c in password):  # +1
            return False

        return True

    def _generate_token(self, username: str) -> str:
        """Generate authentication token."""
        import hashlib
        import time

        data = f"{username}:{self.secret_key}:{time.time()}"
        return hashlib.sha256(data.encode()).hexdigest()


if __name__ == "__main__":
    # Test code
    user = User("John Doe", 25, "john@example.com")
    print(f"User: {user.display_name}, Adult: {user.is_adult}")

    auth = AuthenticationManager("secret123")
    token = auth.authenticate("john", "Password123", remember_me=True)
    print(f"Token: {token}")
