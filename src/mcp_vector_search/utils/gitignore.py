"""Gitignore parsing and matching utilities."""

import fnmatch
import re
from pathlib import Path
from typing import List, Set

from loguru import logger


class GitignorePattern:
    """Represents a single gitignore pattern with its matching logic."""
    
    def __init__(self, pattern: str, is_negation: bool = False, is_directory_only: bool = False):
        """Initialize a gitignore pattern.
        
        Args:
            pattern: The pattern string
            is_negation: Whether this is a negation pattern (starts with !)
            is_directory_only: Whether this pattern only matches directories (ends with /)
        """
        self.original_pattern = pattern
        self.is_negation = is_negation
        self.is_directory_only = is_directory_only
        self.pattern = self._normalize_pattern(pattern)
        
    def _normalize_pattern(self, pattern: str) -> str:
        """Normalize the pattern for matching."""
        # Remove leading ! for negation patterns
        if pattern.startswith('!'):
            pattern = pattern[1:]
            
        # Remove trailing / for directory-only patterns
        if pattern.endswith('/'):
            pattern = pattern[:-1]
            
        # Handle leading slash (absolute from repo root)
        if pattern.startswith('/'):
            pattern = pattern[1:]
            
        return pattern
        
    def matches(self, path: str, is_directory: bool = False) -> bool:
        """Check if this pattern matches the given path.
        
        Args:
            path: Relative path from repository root
            is_directory: Whether the path is a directory
            
        Returns:
            True if the pattern matches
        """
        # Directory-only patterns only match directories
        if self.is_directory_only and not is_directory:
            return False
            
        # Convert path separators for consistent matching
        path = path.replace('\\', '/')
        pattern = self.pattern.replace('\\', '/')
        
        # Try exact match first
        if fnmatch.fnmatch(path, pattern):
            return True
            
        # Try matching any parent directory
        path_parts = path.split('/')
        for i in range(len(path_parts)):
            subpath = '/'.join(path_parts[i:])
            if fnmatch.fnmatch(subpath, pattern):
                return True
                
        # Try matching with ** patterns (glob-style)
        if '**' in pattern:
            # Convert ** to regex pattern
            regex_pattern = pattern.replace('**', '.*')
            regex_pattern = regex_pattern.replace('*', '[^/]*')
            regex_pattern = regex_pattern.replace('?', '[^/]')
            regex_pattern = f'^{regex_pattern}$'
            
            try:
                if re.match(regex_pattern, path):
                    return True
            except re.error:
                # Fallback to simple fnmatch if regex fails
                pass
                
        return False


class GitignoreParser:
    """Parser for .gitignore files with proper pattern matching."""
    
    def __init__(self, project_root: Path):
        """Initialize gitignore parser.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.patterns: List[GitignorePattern] = []
        self._load_gitignore_files()
        
    def _load_gitignore_files(self) -> None:
        """Load all .gitignore files in the project hierarchy."""
        # Load global .gitignore first (if exists)
        global_gitignore = self.project_root / '.gitignore'
        if global_gitignore.exists():
            self._parse_gitignore_file(global_gitignore)
            
        # Load .gitignore files in subdirectories
        for gitignore_file in self.project_root.rglob('.gitignore'):
            if gitignore_file != global_gitignore:
                self._parse_gitignore_file(gitignore_file)
                
    def _parse_gitignore_file(self, gitignore_path: Path) -> None:
        """Parse a single .gitignore file.
        
        Args:
            gitignore_path: Path to the .gitignore file
        """
        try:
            with open(gitignore_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                    
                # Check for negation pattern
                is_negation = line.startswith('!')
                
                # Check for directory-only pattern
                is_directory_only = line.endswith('/')
                
                # Create pattern relative to the .gitignore file's directory
                gitignore_dir = gitignore_path.parent
                if gitignore_dir != self.project_root:
                    # Adjust pattern for subdirectory .gitignore files
                    relative_dir = gitignore_dir.relative_to(self.project_root)
                    if not line.startswith('/') and not is_negation:
                        line = str(relative_dir / line)
                    elif is_negation and not line[1:].startswith('/'):
                        line = '!' + str(relative_dir / line[1:])
                        
                pattern = GitignorePattern(line, is_negation, is_directory_only)
                self.patterns.append(pattern)
                
        except Exception as e:
            logger.warning(f"Failed to parse {gitignore_path}: {e}")
            
    def is_ignored(self, path: Path) -> bool:
        """Check if a path should be ignored according to .gitignore rules.
        
        Args:
            path: Path to check (can be absolute or relative to project root)
            
        Returns:
            True if the path should be ignored
        """
        try:
            # Convert to relative path from project root
            if path.is_absolute():
                relative_path = path.relative_to(self.project_root)
            else:
                relative_path = path
                
            path_str = str(relative_path).replace('\\', '/')
            is_directory = path.is_dir() if path.exists() else False
            
            # Apply patterns in order, with later patterns overriding earlier ones
            ignored = False
            
            for pattern in self.patterns:
                if pattern.matches(path_str, is_directory):
                    ignored = not pattern.is_negation
                    
            return ignored
            
        except ValueError:
            # Path is not relative to project root
            return False
        except Exception as e:
            logger.debug(f"Error checking gitignore for {path}: {e}")
            return False
            
    def get_ignored_patterns(self) -> List[str]:
        """Get list of all ignore patterns.
        
        Returns:
            List of pattern strings
        """
        return [p.original_pattern for p in self.patterns if not p.is_negation]
        
    def get_negation_patterns(self) -> List[str]:
        """Get list of all negation patterns.
        
        Returns:
            List of negation pattern strings
        """
        return [p.original_pattern for p in self.patterns if p.is_negation]


def create_gitignore_parser(project_root: Path) -> GitignoreParser:
    """Create a gitignore parser for the given project.
    
    Args:
        project_root: Root directory of the project
        
    Returns:
        GitignoreParser instance
    """
    return GitignoreParser(project_root)


def is_path_gitignored(path: Path, project_root: Path) -> bool:
    """Quick function to check if a path is gitignored.
    
    Args:
        path: Path to check
        project_root: Root directory of the project
        
    Returns:
        True if the path should be ignored
    """
    parser = create_gitignore_parser(project_root)
    return parser.is_ignored(path)
