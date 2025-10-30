"""Directory index for tracking project structure and file relationships."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from loguru import logger

from .models import Directory


class DirectoryIndex:
    """Manages directory structure and file-directory relationships."""

    def __init__(self, index_path: Path) -> None:
        """Initialize directory index.

        Args:
            index_path: Path to directory index file (JSON)
        """
        self.index_path = index_path
        self.directories: dict[str, Directory] = {}  # path -> Directory
        self.file_to_directory: dict[str, str] = {}  # file_path -> directory_path
        self.directory_files: dict[str, list[str]] = defaultdict(
            list
        )  # dir_path -> [file_paths]

    def load(self) -> None:
        """Load directory index from disk."""
        if not self.index_path.exists():
            logger.debug("No directory index found, starting fresh")
            return

        try:
            with open(self.index_path) as f:
                data = json.load(f)

            # Load directories
            for dir_data in data.get("directories", []):
                directory = Directory.from_dict(dir_data)
                self.directories[str(directory.path)] = directory

            # Load file mappings
            self.file_to_directory = data.get("file_to_directory", {})

            # Rebuild directory_files from file_to_directory
            self.directory_files = defaultdict(list)
            for file_path, dir_path in self.file_to_directory.items():
                self.directory_files[dir_path].append(file_path)

            logger.info(f"Loaded {len(self.directories)} directories from index")

        except Exception as e:
            logger.error(f"Failed to load directory index: {e}")
            self.directories = {}
            self.file_to_directory = {}
            self.directory_files = defaultdict(list)

    def save(self) -> None:
        """Save directory index to disk."""
        try:
            # Ensure parent directory exists
            self.index_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "directories": [d.to_dict() for d in self.directories.values()],
                "file_to_directory": self.file_to_directory,
            }

            with open(self.index_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved {len(self.directories)} directories to index")

        except Exception as e:
            logger.error(f"Failed to save directory index: {e}")
            raise

    def add_directory(self, directory: Directory) -> None:
        """Add or update a directory in the index.

        Args:
            directory: Directory to add
        """
        dir_path = str(directory.path)
        self.directories[dir_path] = directory

    def add_file(self, file_path: Path, directory_path: Path) -> None:
        """Associate a file with its directory.

        Args:
            file_path: Path to the file
            directory_path: Path to the directory containing the file
        """
        file_path_str = str(file_path)
        dir_path_str = str(directory_path)

        self.file_to_directory[file_path_str] = dir_path_str
        if file_path_str not in self.directory_files[dir_path_str]:
            self.directory_files[dir_path_str].append(file_path_str)

        # Update directory file count
        if dir_path_str in self.directories:
            self.directories[dir_path_str].file_count = len(
                self.directory_files[dir_path_str]
            )

    def get_directory(self, directory_path: Path) -> Directory | None:
        """Get directory by path.

        Args:
            directory_path: Path to directory

        Returns:
            Directory object or None if not found
        """
        return self.directories.get(str(directory_path))

    def get_files_in_directory(self, directory_path: Path) -> list[str]:
        """Get all files in a directory.

        Args:
            directory_path: Path to directory

        Returns:
            List of file paths in the directory
        """
        return self.directory_files.get(str(directory_path), [])

    def get_subdirectories(self, directory_path: Path) -> list[Directory]:
        """Get all immediate subdirectories.

        Args:
            directory_path: Path to parent directory

        Returns:
            List of subdirectory objects
        """
        parent_path_str = str(directory_path)
        subdirs = []

        for _dir_path_str, directory in self.directories.items():
            if directory.parent_path and str(directory.parent_path) == parent_path_str:
                subdirs.append(directory)

        return subdirs

    def get_root_directories(self) -> list[Directory]:
        """Get all root-level directories (no parent).

        Returns:
            List of root directory objects
        """
        return [d for d in self.directories.values() if d.parent_path is None]

    def delete_directory(self, directory_path: Path) -> None:
        """Remove directory and its file associations.

        Args:
            directory_path: Path to directory to remove
        """
        dir_path_str = str(directory_path)

        # Remove directory
        if dir_path_str in self.directories:
            del self.directories[dir_path_str]

        # Remove file associations
        if dir_path_str in self.directory_files:
            for file_path in self.directory_files[dir_path_str]:
                if file_path in self.file_to_directory:
                    del self.file_to_directory[file_path]
            del self.directory_files[dir_path_str]

    def delete_file(self, file_path: Path) -> None:
        """Remove file from directory associations.

        Args:
            file_path: Path to file to remove
        """
        file_path_str = str(file_path)

        if file_path_str in self.file_to_directory:
            dir_path = self.file_to_directory[file_path_str]
            del self.file_to_directory[file_path_str]

            # Remove from directory_files
            if dir_path in self.directory_files:
                self.directory_files[dir_path] = [
                    f for f in self.directory_files[dir_path] if f != file_path_str
                ]

                # Update directory file count
                if dir_path in self.directories:
                    self.directories[dir_path].file_count = len(
                        self.directory_files[dir_path]
                    )

    def rebuild_from_files(
        self,
        file_paths: list[Path],
        root_path: Path,
        chunk_stats: dict[str, dict] | None = None,
    ) -> None:
        """Rebuild directory index from list of files with statistics from chunks.

        Args:
            file_paths: List of file paths to index
            root_path: Project root path
            chunk_stats: Optional dict mapping file_path -> {'chunks': count, 'language': str}
        """
        self.directories = {}
        self.file_to_directory = {}
        self.directory_files = defaultdict(list)

        # Track all unique directories and their statistics
        dir_set = set()
        dir_chunks = defaultdict(int)  # directory -> total chunks
        dir_languages = defaultdict(
            lambda: defaultdict(int)
        )  # directory -> {language: count}
        dir_modified = defaultdict(float)  # directory -> most recent modification time

        for file_path in file_paths:
            # Get relative path from root
            try:
                rel_path = file_path.relative_to(root_path)
                parent_dir = rel_path.parent

                # Add all parent directories up to root
                current = parent_dir
                while current != Path("."):
                    dir_set.add(current)

                    # Accumulate statistics up the directory tree
                    if chunk_stats and str(file_path) in chunk_stats:
                        stats = chunk_stats[str(file_path)]
                        dir_chunks[current] += stats.get("chunks", 0)
                        if "language" in stats:
                            dir_languages[current][stats["language"]] += stats.get(
                                "chunks", 0
                            )
                        # Track most recent modification time
                        if "modified" in stats:
                            dir_modified[current] = max(
                                dir_modified.get(current, 0), stats["modified"]
                            )

                    current = current.parent

                # Associate file with its direct parent
                if parent_dir != Path("."):
                    self.add_file(rel_path, parent_dir)

            except ValueError:
                # File not relative to root, skip
                logger.warning(f"File {file_path} not under root {root_path}")
                continue

        # Create Directory objects for all directories
        for dir_path in sorted(dir_set):
            # Determine parent
            parent_path = dir_path.parent if dir_path.parent != Path(".") else None

            # Check if it's a package
            is_package = False
            full_dir_path = root_path / dir_path
            if (full_dir_path / "__init__.py").exists():
                is_package = True
            elif (full_dir_path / "package.json").exists():
                is_package = True

            directory = Directory(
                path=dir_path,
                name=dir_path.name,
                parent_path=parent_path,
                depth=len(dir_path.parts),
                is_package=is_package,
                total_chunks=dir_chunks.get(dir_path, 0),
                languages=dict(dir_languages.get(dir_path, {})),
                last_modified=dir_modified.get(dir_path),
            )

            self.add_directory(directory)

        # Update subdirectory counts
        for directory in self.directories.values():
            subdirs = self.get_subdirectories(directory.path)
            directory.subdirectory_count = len(subdirs)

        logger.info(
            f"Rebuilt directory index with {len(self.directories)} directories, {sum(dir_chunks.values())} total chunks"
        )

    def get_stats(self) -> dict[str, Any]:
        """Get directory index statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_directories": len(self.directories),
            "total_files": len(self.file_to_directory),
            "root_directories": len(self.get_root_directories()),
            "packages": sum(1 for d in self.directories.values() if d.is_package),
        }

    def reset(self) -> None:
        """Clear all directory data."""
        self.directories = {}
        self.file_to_directory = {}
        self.directory_files = defaultdict(list)

        if self.index_path.exists():
            self.index_path.unlink()

        logger.info("Directory index reset")
