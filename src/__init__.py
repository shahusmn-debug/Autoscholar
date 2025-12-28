# AutoScholar-MVP Source Package
"""
AutoScholar-MVP: Autonomous Research Agent

This package contains the core modules for the research agent:
- client: Gemini API wrapper
- state: State management schemas
- ingestion: Data and reference ingestion
- tools: Python sandbox for code execution
- progress: CLI progress and logging
- graph: LangGraph state machine
"""

from .state import PaperState

__version__ = "0.1.0"
__all__ = ["PaperState"]
