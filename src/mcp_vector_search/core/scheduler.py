"""Scheduling utilities for automatic reindexing."""

import platform
import subprocess
import sys
from pathlib import Path

from loguru import logger


class SchedulerManager:
    """Manages scheduled tasks for automatic reindexing."""

    def __init__(self, project_root: Path):
        """Initialize scheduler manager.

        Args:
            project_root: Project root directory
        """
        self.project_root = project_root
        self.system = platform.system().lower()

    def install_scheduled_task(
        self, interval_minutes: int = 60, task_name: str | None = None
    ) -> bool:
        """Install a scheduled task for automatic reindexing.

        Args:
            interval_minutes: Interval between reindex checks in minutes
            task_name: Custom task name (auto-generated if None)

        Returns:
            True if task was installed successfully
        """
        if task_name is None:
            safe_path = str(self.project_root).replace("/", "_").replace("\\", "_")
            task_name = f"mcp_vector_search_reindex_{safe_path}"

        if self.system == "linux" or self.system == "darwin":
            return self._install_cron_job(interval_minutes, task_name)
        elif self.system == "windows":
            return self._install_windows_task(interval_minutes, task_name)
        else:
            logger.error(f"Unsupported system: {self.system}")
            return False

    def uninstall_scheduled_task(self, task_name: str | None = None) -> bool:
        """Uninstall scheduled task.

        Args:
            task_name: Task name to uninstall (auto-generated if None)

        Returns:
            True if task was uninstalled successfully
        """
        if task_name is None:
            safe_path = str(self.project_root).replace("/", "_").replace("\\", "_")
            task_name = f"mcp_vector_search_reindex_{safe_path}"

        if self.system == "linux" or self.system == "darwin":
            return self._uninstall_cron_job(task_name)
        elif self.system == "windows":
            return self._uninstall_windows_task(task_name)
        else:
            logger.error(f"Unsupported system: {self.system}")
            return False

    def _install_cron_job(self, interval_minutes: int, task_name: str) -> bool:
        """Install cron job on Linux/macOS."""
        try:
            # Generate cron command
            python_path = sys.executable
            project_root = str(self.project_root)

            # Create wrapper script
            script_content = f"""#!/bin/bash
# MCP Vector Search Auto-Reindex - {task_name}
cd "{project_root}" || exit 1

# Check if mcp-vector-search is available
if command -v mcp-vector-search &> /dev/null; then
    mcp-vector-search auto-index check --auto-reindex --max-files 10
elif [ -f "{python_path}" ]; then
    "{python_path}" -m mcp_vector_search auto-index check --auto-reindex --max-files 10
else
    python3 -m mcp_vector_search auto-index check --auto-reindex --max-files 10
fi
"""

            # Write script to temp file
            script_dir = Path.home() / ".mcp-vector-search" / "scripts"
            script_dir.mkdir(parents=True, exist_ok=True)
            script_file = script_dir / f"{task_name}.sh"

            script_file.write_text(script_content)
            script_file.chmod(0o755)

            # Calculate cron schedule
            if interval_minutes >= 60:
                # Hourly or less frequent
                hours = interval_minutes // 60
                cron_schedule = f"0 */{hours} * * *"
            else:
                # More frequent than hourly
                cron_schedule = f"*/{interval_minutes} * * * *"

            # Add to crontab
            cron_entry = f"{cron_schedule} {script_file} # {task_name}\n"

            # Get current crontab
            try:
                result = subprocess.run(  # nosec B607
                    ["crontab", "-l"], capture_output=True, text=True, check=True
                )
                current_crontab = result.stdout
            except subprocess.CalledProcessError:
                current_crontab = ""

            # Check if entry already exists
            if task_name in current_crontab:
                logger.info(f"Cron job {task_name} already exists")
                return True

            # Add new entry
            new_crontab = current_crontab + cron_entry

            # Install new crontab
            process = subprocess.Popen(  # nosec B607
                ["crontab", "-"], stdin=subprocess.PIPE, text=True
            )
            process.communicate(input=new_crontab)

            if process.returncode == 0:
                logger.info(f"Installed cron job: {task_name}")
                logger.info(f"Schedule: every {interval_minutes} minutes")
                logger.info(f"Script: {script_file}")
                return True
            else:
                logger.error("Failed to install cron job")
                return False

        except Exception as e:
            logger.error(f"Failed to install cron job: {e}")
            return False

    def _uninstall_cron_job(self, task_name: str) -> bool:
        """Uninstall cron job on Linux/macOS."""
        try:
            # Get current crontab
            try:
                result = subprocess.run(  # nosec B607
                    ["crontab", "-l"], capture_output=True, text=True, check=True
                )
                current_crontab = result.stdout
            except subprocess.CalledProcessError:
                logger.info("No crontab found")
                return True

            # Remove lines containing task name
            lines = current_crontab.split("\n")
            new_lines = [line for line in lines if task_name not in line]
            new_crontab = "\n".join(new_lines)

            # Install new crontab
            if new_crontab.strip():
                process = subprocess.Popen(  # nosec B607
                    ["crontab", "-"], stdin=subprocess.PIPE, text=True
                )
                process.communicate(input=new_crontab)
            else:
                # Remove crontab entirely if empty
                subprocess.run(["crontab", "-r"], check=False)  # nosec B607

            # Remove script file
            script_dir = Path.home() / ".mcp-vector-search" / "scripts"
            script_file = script_dir / f"{task_name}.sh"
            if script_file.exists():
                script_file.unlink()

            logger.info(f"Uninstalled cron job: {task_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to uninstall cron job: {e}")
            return False

    def _install_windows_task(self, interval_minutes: int, task_name: str) -> bool:
        """Install Windows scheduled task."""
        try:
            python_path = sys.executable
            project_root = str(self.project_root)

            # Create PowerShell script
            script_content = f"""# MCP Vector Search Auto-Reindex - {task_name}
Set-Location "{project_root}"

try {{
    if (Get-Command "mcp-vector-search" -ErrorAction SilentlyContinue) {{
        mcp-vector-search auto-index check --auto-reindex --max-files 10
    }} elseif (Test-Path "{python_path}") {{
        & "{python_path}" -m mcp_vector_search auto-index check --auto-reindex --max-files 10
    }} else {{
        python -m mcp_vector_search auto-index check --auto-reindex --max-files 10
    }}
}} catch {{
    # Silently ignore errors
}}
"""

            # Write script
            script_dir = Path.home() / ".mcp-vector-search" / "scripts"
            script_dir.mkdir(parents=True, exist_ok=True)
            script_file = script_dir / f"{task_name}.ps1"

            script_file.write_text(script_content)

            # Create scheduled task using schtasks
            cmd = [
                "schtasks",
                "/create",
                "/tn",
                task_name,
                "/tr",
                f'powershell.exe -ExecutionPolicy Bypass -File "{script_file}"',
                "/sc",
                "minute",
                "/mo",
                str(interval_minutes),
                "/f",  # Force overwrite if exists
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"Installed Windows task: {task_name}")
                logger.info(f"Schedule: every {interval_minutes} minutes")
                return True
            else:
                logger.error(f"Failed to install Windows task: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Failed to install Windows task: {e}")
            return False

    def _uninstall_windows_task(self, task_name: str) -> bool:
        """Uninstall Windows scheduled task."""
        try:
            # Delete scheduled task
            cmd = ["schtasks", "/delete", "/tn", task_name, "/f"]
            result = subprocess.run(cmd, capture_output=True, text=True)

            # Remove script file
            script_dir = Path.home() / ".mcp-vector-search" / "scripts"
            script_file = script_dir / f"{task_name}.ps1"
            if script_file.exists():
                script_file.unlink()

            if result.returncode == 0:
                logger.info(f"Uninstalled Windows task: {task_name}")
                return True
            else:
                # Task might not exist, which is fine
                logger.info(
                    f"Windows task {task_name} was not found (already uninstalled)"
                )
                return True

        except Exception as e:
            logger.error(f"Failed to uninstall Windows task: {e}")
            return False

    def get_scheduled_task_status(self, task_name: str | None = None) -> dict:
        """Get status of scheduled tasks.

        Args:
            task_name: Task name to check (auto-generated if None)

        Returns:
            Dictionary with task status information
        """
        if task_name is None:
            safe_path = str(self.project_root).replace("/", "_").replace("\\", "_")
            task_name = f"mcp_vector_search_reindex_{safe_path}"

        status = {
            "system": self.system,
            "task_name": task_name,
            "exists": False,
            "enabled": False,
            "last_run": None,
            "next_run": None,
        }

        if self.system == "linux" or self.system == "darwin":
            status.update(self._get_cron_status(task_name))
        elif self.system == "windows":
            status.update(self._get_windows_task_status(task_name))

        return status

    def _get_cron_status(self, task_name: str) -> dict:
        """Get cron job status."""
        try:
            result = subprocess.run(  # nosec B607
                ["crontab", "-l"], capture_output=True, text=True, check=True
            )

            exists = task_name in result.stdout
            return {"exists": exists, "enabled": exists}

        except subprocess.CalledProcessError:
            return {"exists": False, "enabled": False}

    def _get_windows_task_status(self, task_name: str) -> dict:
        """Get Windows task status."""
        try:
            result = subprocess.run(  # nosec B607
                ["schtasks", "/query", "/tn", task_name], capture_output=True, text=True
            )

            if result.returncode == 0:
                # Parse output for status
                enabled = "Ready" in result.stdout or "Running" in result.stdout
                return {"exists": True, "enabled": enabled}
            else:
                return {"exists": False, "enabled": False}

        except Exception:
            return {"exists": False, "enabled": False}
