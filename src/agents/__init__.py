# AutoScholar-MVP Agents Package
"""
Agent modules for the research pipeline:
- analyst: Data analysis and visualization
- writer: Paper authoring with dual personas
- reviewer: Draft scoring and feedback
"""

from .analyst import node_analyst
from .writer import node_writer
from .reviewer import node_reviewer

__all__ = ["node_analyst", "node_writer", "node_reviewer"]
