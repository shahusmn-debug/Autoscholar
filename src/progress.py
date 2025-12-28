# src/progress.py
"""
Progress Tracking and Logging

This module provides:
- Real-time CLI progress display using Rich
- Dual-file logging (human-readable + machine-parseable)
- Event tracking for debugging and analysis
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.live import Live


# Global state for the progress tracker
_console = Console()
_run_dir: Optional[Path] = None
_progress: Optional[Progress] = None
_live: Optional[Live] = None
_current_task_id = None


def start_progress(run_dir: str) -> None:
    """
    Initialize the progress tracking system.
    
    Creates the log files and starts the live display.
    
    Args:
        run_dir: Path to the current run's output directory
    """
    global _run_dir, _progress, _live, _current_task_id
    
    _run_dir = Path(run_dir)
    _run_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize log files
    (run_dir.parent / "run.log" if isinstance(run_dir, Path) else Path(run_dir) / "run.log").touch()
    (run_dir.parent / "events.jsonl" if isinstance(run_dir, Path) else Path(run_dir) / "events.jsonl").touch()
    
    # Create Rich progress display
    _progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        TextColumn("[dim]{task.fields[detail]}"),
        TimeElapsedColumn(),
        console=_console,
    )
    
    # Start with initial task
    _current_task_id = _progress.add_task(
        "Initializing...",
        detail="Setting up pipeline",
        total=None  # Indeterminate
    )
    
    _live = Live(_progress, console=_console, refresh_per_second=4)
    _live.start()
    
    log_event("INFO", "Setup", f"Run started at {run_dir}")


def stop_progress() -> None:
    """Stop the live progress display."""
    global _live, _progress
    
    if _live:
        _live.stop()
        _live = None
    _progress = None


def log_event(
    level: str,
    stage: str,
    message: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log an event to both human and machine log files.
    
    This is the primary logging function. It writes a human-readable
    line to run.log and a JSON object to events.jsonl.
    
    Args:
        level: Log level (INFO, WARNING, ERROR, DEBUG)
        stage: Pipeline stage (Setup, Ingestion, Analyst, Writer, Reviewer)
        message: Human-readable message
        metadata: Optional dictionary of additional data for events.jsonl
    """
    global _run_dir
    
    if _run_dir is None:
        # Fallback to console if not initialized
        _console.print(f"[{level}] [{stage}] {message}")
        return
    
    timestamp = datetime.now()
    time_str = timestamp.strftime("%H:%M:%S")
    
    # Write to human log (run.log)
    human_log = _run_dir / "run.log"
    with open(human_log, "a", encoding="utf-8") as f:
        f.write(f"[{time_str}] {level} [{stage}] {message}\n")
    
    # Write to machine log (events.jsonl)
    event = {
        "timestamp": timestamp.isoformat(),
        "level": level,
        "stage": stage,
        "message": message,
    }
    if metadata:
        event["metadata"] = metadata
    
    machine_log = _run_dir / "events.jsonl"
    with open(machine_log, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def update_display(stage: str, detail: str) -> None:
    """
    Update the live CLI spinner with current status.
    
    Args:
        stage: Current pipeline stage (shown in bold)
        detail: Additional detail (shown dimmed)
    """
    global _progress, _current_task_id
    
    if _progress and _current_task_id is not None:
        _progress.update(
            _current_task_id,
            description=stage,
            detail=detail
        )


def print_header(title: str) -> None:
    """Print a styled header panel."""
    _console.print(Panel(f"[bold cyan]{title}[/]", expand=False))


def print_success(message: str) -> None:
    """Print a success message in green."""
    _console.print(f"[bold green]✓[/] {message}")


def print_error(message: str) -> None:
    """Print an error message in red."""
    _console.print(f"[bold red]✗[/] {message}")


def print_warning(message: str) -> None:
    """Print a warning message in yellow."""
    _console.print(f"[bold yellow]![/] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    _console.print(f"[dim]→[/] {message}")
