"""Unit tests for FileWatcher and CodeFileHandler."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_vector_search.config.settings import ProjectConfig
from mcp_vector_search.core.watcher import CodeFileHandler, FileWatcher

# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------


def _make_event(
    src_path: str, is_directory: bool = False, dest_path: str | None = None
):
    """Build a minimal watchdog-style event object."""
    event = MagicMock()
    event.src_path = src_path
    event.is_directory = is_directory
    if dest_path is not None:
        event.dest_path = dest_path
    else:
        # Simulate events that have no dest_path (on_modified / on_created / on_deleted)
        del event.dest_path
    return event


def _make_handler(
    file_extensions: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
    loop: asyncio.AbstractEventLoop | None = None,
    debounce_delay: float = 0.05,
) -> tuple[CodeFileHandler, AsyncMock]:
    """Create a CodeFileHandler with a mock callback and return both."""
    callback = AsyncMock()
    handler = CodeFileHandler(
        file_extensions=file_extensions or [".py", ".js"],
        ignore_patterns=ignore_patterns or [".git", ".mcp-vector-search"],
        callback=callback,
        loop=loop or asyncio.get_event_loop(),
        debounce_delay=debounce_delay,
    )
    return handler, callback


def _make_watcher(
    tmp_path: Path,
    file_extensions: list[str] | None = None,
) -> tuple[FileWatcher, MagicMock, AsyncMock]:
    """Create a FileWatcher with mocked indexer and database."""
    config = ProjectConfig(
        project_root=tmp_path,
        index_path=tmp_path / ".mcp-vector-search",
        file_extensions=file_extensions or [".py"],
    )
    indexer = MagicMock()
    indexer.index_file = AsyncMock(return_value=5)
    database = MagicMock()
    database.delete_by_file = AsyncMock(return_value=3)
    watcher = FileWatcher(
        project_root=tmp_path,
        config=config,
        indexer=indexer,
        database=database,
    )
    return watcher, indexer, database


# ---------------------------------------------------------------------------
# FileWatcher lifecycle
# ---------------------------------------------------------------------------


class TestFileWatcherLifecycle:
    """Tests for FileWatcher start/stop lifecycle."""

    async def test_start_sets_is_running_true(self, tmp_path: Path):
        """start() should set is_running = True and start the observer."""
        watcher, _, _ = _make_watcher(tmp_path)

        with patch("mcp_vector_search.core.watcher.Observer") as mock_observer_cls:
            mock_obs = MagicMock()
            mock_observer_cls.return_value = mock_obs

            await watcher.start()

        assert watcher.is_running is True
        mock_obs.start.assert_called_once()

    async def test_stop_sets_is_running_false(self, tmp_path: Path):
        """stop() should set is_running = False and stop the observer."""
        watcher, _, _ = _make_watcher(tmp_path)

        with patch("mcp_vector_search.core.watcher.Observer") as mock_observer_cls:
            mock_obs = MagicMock()
            mock_observer_cls.return_value = mock_obs

            await watcher.start()
            assert watcher.is_running is True

            await watcher.stop()

        assert watcher.is_running is False
        mock_obs.stop.assert_called_once()
        mock_obs.join.assert_called_once()

    async def test_stop_is_idempotent(self, tmp_path: Path):
        """stop() called on a watcher that is not running should be a no-op."""
        watcher, _, _ = _make_watcher(tmp_path)
        assert watcher.is_running is False

        # Should not raise
        await watcher.stop()

        assert watcher.is_running is False

    async def test_start_twice_is_idempotent(self, tmp_path: Path):
        """Calling start() when already running should be a no-op (no second observer)."""
        watcher, _, _ = _make_watcher(tmp_path)

        with patch("mcp_vector_search.core.watcher.Observer") as mock_observer_cls:
            mock_obs = MagicMock()
            mock_observer_cls.return_value = mock_obs

            await watcher.start()
            await watcher.start()  # Second call should be ignored

        # Observer constructor + start should only have been called once
        assert mock_observer_cls.call_count == 1
        mock_obs.start.assert_called_once()


# ---------------------------------------------------------------------------
# CodeFileHandler.should_process_file
# ---------------------------------------------------------------------------


class TestShouldProcessFile:
    """Tests for the file-filter logic."""

    def test_returns_true_for_known_extension(self):
        handler, _ = _make_handler(file_extensions=[".py"])
        assert handler.should_process_file("/project/src/foo.py") is True

    def test_returns_false_for_unknown_extension(self):
        handler, _ = _make_handler(file_extensions=[".py"])
        assert handler.should_process_file("/project/logs/app.log") is False
        assert handler.should_process_file("/project/notes.txt") is False

    def test_returns_false_for_ignored_pattern(self):
        handler, _ = _make_handler(
            file_extensions=[".py"],
            ignore_patterns=[".mcp-vector-search"],
        )
        assert (
            handler.should_process_file("/project/.mcp-vector-search/data.py") is False
        )

    def test_returns_false_for_git_directory(self):
        handler, _ = _make_handler(
            file_extensions=[".py"],
            ignore_patterns=[".git"],
        )
        assert handler.should_process_file("/project/.git/COMMIT_EDITMSG") is False


# ---------------------------------------------------------------------------
# CodeFileHandler — event routing
# ---------------------------------------------------------------------------


class TestCodeFileHandlerEventRouting:
    """Tests verifying which events are routed to _schedule_change."""

    def _capture_schedules(self, handler: CodeFileHandler) -> list[tuple[str, str]]:
        """Patch _schedule_change to record calls."""
        calls: list[tuple[str, str]] = []

        def _mock_schedule(file_path: str, change_type: str) -> None:
            calls.append((file_path, change_type))

        handler._schedule_change = _mock_schedule
        return calls

    # --- on_modified ---

    def test_on_modified_py_file_schedules_change(self):
        handler, _ = _make_handler(file_extensions=[".py"])
        calls = self._capture_schedules(handler)

        event = _make_event("/project/src/module.py")
        handler.on_modified(event)

        assert len(calls) == 1
        assert calls[0] == ("/project/src/module.py", "modified")

    def test_on_modified_non_code_file_is_ignored(self):
        handler, _ = _make_handler(file_extensions=[".py"])
        calls = self._capture_schedules(handler)

        for path in ("/project/app.log", "/project/README.txt"):
            handler.on_modified(_make_event(path))

        assert calls == []

    def test_on_modified_directory_event_is_ignored(self):
        handler, _ = _make_handler(file_extensions=[".py"])
        calls = self._capture_schedules(handler)

        event = _make_event("/project/src", is_directory=True)
        handler.on_modified(event)

        assert calls == []

    # --- on_created ---

    def test_on_created_py_file_schedules_change(self):
        handler, _ = _make_handler(file_extensions=[".py"])
        calls = self._capture_schedules(handler)

        event = _make_event("/project/new_module.py")
        handler.on_created(event)

        assert len(calls) == 1
        assert calls[0] == ("/project/new_module.py", "created")

    def test_on_created_non_code_file_is_ignored(self):
        handler, _ = _make_handler(file_extensions=[".py"])
        calls = self._capture_schedules(handler)

        handler.on_created(_make_event("/project/notes.txt"))

        assert calls == []

    # --- on_deleted ---

    def test_on_deleted_py_file_schedules_change(self):
        handler, _ = _make_handler(file_extensions=[".py"])
        calls = self._capture_schedules(handler)

        event = _make_event("/project/old_module.py")
        handler.on_deleted(event)

        assert len(calls) == 1
        assert calls[0] == ("/project/old_module.py", "deleted")

    # --- on_moved ---

    def test_on_moved_py_to_py_schedules_delete_and_create(self):
        handler, _ = _make_handler(file_extensions=[".py"])
        calls = self._capture_schedules(handler)

        event = _make_event(
            src_path="/project/old.py",
            dest_path="/project/new.py",
        )
        handler.on_moved(event)

        assert len(calls) == 2
        assert ("/project/old.py", "deleted") in calls
        assert ("/project/new.py", "created") in calls

    def test_on_moved_py_to_txt_schedules_only_delete(self):
        """Rename from .py to .txt: old file deleted, new file not a code file."""
        handler, _ = _make_handler(file_extensions=[".py"])
        calls = self._capture_schedules(handler)

        event = _make_event(
            src_path="/project/module.py",
            dest_path="/project/module.txt",
        )
        handler.on_moved(event)

        assert calls == [("/project/module.py", "deleted")]

    def test_on_moved_txt_to_py_schedules_only_create(self):
        """Rename from .txt to .py: source not a code file, destination is."""
        handler, _ = _make_handler(file_extensions=[".py"])
        calls = self._capture_schedules(handler)

        event = _make_event(
            src_path="/project/notes.txt",
            dest_path="/project/notes.py",
        )
        handler.on_moved(event)

        assert calls == [("/project/notes.py", "created")]

    # --- ignore patterns ---

    def test_mcp_vector_search_directory_events_ignored(self):
        handler, _ = _make_handler(
            file_extensions=[".py"],
            ignore_patterns=[".mcp-vector-search"],
        )
        calls = self._capture_schedules(handler)

        for method in (handler.on_modified, handler.on_created, handler.on_deleted):
            method(_make_event("/project/.mcp-vector-search/index.py"))

        assert calls == []

    def test_git_directory_events_ignored(self):
        handler, _ = _make_handler(
            file_extensions=[".py"],
            ignore_patterns=[".git"],
        )
        calls = self._capture_schedules(handler)

        for method in (handler.on_modified, handler.on_created, handler.on_deleted):
            method(_make_event("/project/.git/hooks/pre-commit.py"))

        assert calls == []


# ---------------------------------------------------------------------------
# FileWatcher._remove_file_chunks
# ---------------------------------------------------------------------------


class TestRemoveFileChunks:
    """Tests for _remove_file_chunks."""

    async def test_calls_database_delete_with_relative_path(self, tmp_path: Path):
        """delete_by_file should receive the path relative to project_root."""
        watcher, _, database = _make_watcher(tmp_path)

        file_path = tmp_path / "src" / "module.py"
        await watcher._remove_file_chunks(file_path)

        expected_relative = Path("src/module.py")
        database.delete_by_file.assert_awaited_once_with(expected_relative)

    async def test_handles_value_error_from_relative_to(self, tmp_path: Path):
        """When file_path is not under project_root, fall back to the absolute path."""
        watcher, _, database = _make_watcher(tmp_path)

        outside_path = Path("/some/other/directory/module.py")
        # Should not raise
        await watcher._remove_file_chunks(outside_path)

        # The fallback is to pass the absolute path itself
        database.delete_by_file.assert_awaited_once_with(outside_path)


# ---------------------------------------------------------------------------
# FileWatcher._reindex_file
# ---------------------------------------------------------------------------


class TestReindexFile:
    """Tests for _reindex_file."""

    async def test_existing_file_removes_chunks_then_indexes(self, tmp_path: Path):
        """For an existing file, _remove_file_chunks is called before indexer.index_file."""
        watcher, indexer, database = _make_watcher(tmp_path)

        # Create an actual file so file_path.exists() returns True
        target = tmp_path / "app.py"
        target.write_text("x = 1")

        call_order: list[str] = []
        database.delete_by_file = AsyncMock(
            side_effect=lambda *a, **kw: call_order.append("delete")
        )
        indexer.index_file = AsyncMock(
            side_effect=lambda *a, **kw: call_order.append("index") or 3
        )

        await watcher._reindex_file(target)

        assert "delete" in call_order
        assert "index" in call_order
        assert call_order.index("delete") < call_order.index("index"), (
            "delete_by_file should be called before index_file"
        )
        indexer.index_file.assert_awaited_once_with(target)

    async def test_non_existent_file_logs_warning_no_index(self, tmp_path: Path):
        """For a file that does not exist, indexer.index_file must NOT be called."""
        watcher, indexer, database = _make_watcher(tmp_path)

        missing = tmp_path / "ghost.py"
        assert not missing.exists()

        await watcher._reindex_file(missing)

        indexer.index_file.assert_not_awaited()


# ---------------------------------------------------------------------------
# FileWatcher._handle_file_change — integration of routing
# ---------------------------------------------------------------------------


class TestHandleFileChange:
    """Tests for the _handle_file_change callback used by CodeFileHandler."""

    async def test_deleted_event_calls_remove_file_chunks(self, tmp_path: Path):
        watcher, indexer, database = _make_watcher(tmp_path)

        file_path = str(tmp_path / "removed.py")
        await watcher._handle_file_change(file_path, "deleted")

        database.delete_by_file.assert_awaited_once()
        indexer.index_file.assert_not_awaited()

    async def test_created_event_calls_reindex_file(self, tmp_path: Path):
        watcher, indexer, _ = _make_watcher(tmp_path)

        target = tmp_path / "new_file.py"
        target.write_text("y = 2")

        await watcher._handle_file_change(str(target), "created")

        indexer.index_file.assert_awaited_once_with(target)

    async def test_modified_event_calls_reindex_file(self, tmp_path: Path):
        watcher, indexer, _ = _make_watcher(tmp_path)

        target = tmp_path / "modified.py"
        target.write_text("z = 3")

        await watcher._handle_file_change(str(target), "modified")

        indexer.index_file.assert_awaited_once_with(target)


# ---------------------------------------------------------------------------
# Debounce
# ---------------------------------------------------------------------------


class TestDebounce:
    """Tests for the debounce mechanism in CodeFileHandler."""

    async def test_two_rapid_modifications_result_in_one_callback(self, tmp_path: Path):
        """Two rapid on_modified events for the same file should coalesce into one callback."""
        loop = asyncio.get_running_loop()

        # Use a very short debounce delay so the test runs quickly
        callback = AsyncMock()
        handler = CodeFileHandler(
            file_extensions=[".py"],
            ignore_patterns=[],
            callback=callback,
            loop=loop,
            debounce_delay=0.05,  # 50 ms
        )

        file_path = str(tmp_path / "hot.py")

        # Fire two rapid modify events before the debounce window closes
        handler._schedule_change(file_path, "modified")
        handler._schedule_change(file_path, "modified")

        # Wait long enough for the debounce to fire (debounce_delay + buffer)
        await asyncio.sleep(0.2)

        # The callback should have been invoked exactly once for this file/type pair
        # (pending_changes is a set, so duplicates are collapsed)
        assert callback.await_count == 1
        call_args = callback.call_args
        assert call_args[0][0] == file_path
        assert call_args[0][1] == "modified"

    async def test_different_files_each_get_one_callback(self, tmp_path: Path):
        """Rapid events for two different files should each produce one callback."""
        loop = asyncio.get_running_loop()

        callback = AsyncMock()
        handler = CodeFileHandler(
            file_extensions=[".py"],
            ignore_patterns=[],
            callback=callback,
            loop=loop,
            debounce_delay=0.05,
        )

        path_a = str(tmp_path / "a.py")
        path_b = str(tmp_path / "b.py")

        handler._schedule_change(path_a, "modified")
        handler._schedule_change(path_b, "modified")

        await asyncio.sleep(0.2)

        assert callback.await_count == 2
        invoked_paths = {call[0][0] for call in callback.call_args_list}
        assert path_a in invoked_paths
        assert path_b in invoked_paths


# ---------------------------------------------------------------------------
# Async context manager
# ---------------------------------------------------------------------------


class TestAsyncContextManager:
    """Tests for __aenter__ / __aexit__."""

    async def test_context_manager_starts_and_stops(self, tmp_path: Path):
        watcher, _, _ = _make_watcher(tmp_path)

        with patch("mcp_vector_search.core.watcher.Observer") as mock_observer_cls:
            mock_obs = MagicMock()
            mock_observer_cls.return_value = mock_obs

            async with watcher:
                assert watcher.is_running is True

        assert watcher.is_running is False
        mock_obs.stop.assert_called_once()
