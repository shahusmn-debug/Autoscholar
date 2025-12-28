# src/state.py
"""
State Management - Data Schemas and PaperState

This module defines all the data structures used throughout the
AutoScholar pipeline, including artifact schemas and the global state.
"""

from typing import TypedDict, List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum


class RunMode(str, Enum):
    """The two modes of operation for AutoScholar."""
    CREATE = "create"   # Generate paper from scratch
    REFINE = "refine"   # Improve an existing draft


@dataclass
class EvidenceItem:
    """
    A piece of evidence extracted from raw data.
    
    Evidence items represent factual findings from CSV files,
    data analysis, or other raw data sources.
    """
    evidence_id: str          # Unique identifier (e.g., "ev_001")
    source_file: str          # Original file path
    description: str          # What this evidence shows
    data_snippet: str         # Relevant data excerpt
    statistical_summary: Optional[str] = None


@dataclass
class FigureItem:
    """
    A generated figure/chart from the Analyst.
    
    Figure items link generated visualizations to their
    code, interpretation, and suggested captions.
    """
    figure_id: str            # Unique identifier (e.g., "fig_1")
    file_path: str            # Path to the .png file
    code_path: str            # Path to the generating Python script
    interpretation: str       # Vision model's description
    suggested_caption: str    # Suggested figure caption
    data_source: str          # Which data file this came from


@dataclass
class ReferenceItem:
    """
    A reference from the academic literature.
    
    Reference items store extracted information from PDFs
    and text notes for citation and knowledge retrieval.
    """
    ref_id: str               # Citation key (e.g., "Ref_Smith_2023")
    source_file: str          # Original PDF or text file path
    title: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[str] = None
    summary: str = ""         # Extracted summary/abstract
    key_findings: List[str] = field(default_factory=list)
    methodology: Optional[str] = None
    relevance_notes: str = "" # How this relates to our research


class PaperState(TypedDict, total=False):
    """
    The global state dictionary for the LangGraph pipeline.
    
    This TypedDict defines all fields that flow through the graph.
    Some fields are static (set once during ingestion), while others
    are updated iteratively during the write/review loop.
    
    Static Context:
        knowledge_base: All ingested references keyed by ref_id
        analysis_context: Plain text summary from the Analyst
        figure_manifest: All generated figures keyed by figure_id
        evidence_items: Extracted evidence from raw data
        
    Planning:
        section_queue: Ordered list of sections to write
        current_section: The section currently being worked on
        
    Iteration Memory:
        current_draft: The latest version of the paper/section
        current_iteration: Which iteration we're on (starts at 0 for refine mode)
        previous_score: Score from the last review
        critique_history: List of all feedback received
        
    Metadata:
        mode: "create" or "refine"
        topic: The research topic (for create mode)
        run_dir: Path to this run's output directory
        
    Safety/Recovery:
        best_draft_so_far: Highest-scoring draft seen
        best_score_so_far: Highest score achieved
    """
    
    # --- STATIC CONTEXT (Read-Only after ingestion) ---
    knowledge_base: Dict[str, ReferenceItem]
    analysis_context: str
    figure_manifest: Dict[str, FigureItem]
    evidence_items: Dict[str, EvidenceItem]
    
    # --- PLANNING ---
    section_queue: List[str]
    current_section: str
    
    # --- ITERATION MEMORY ---
    current_draft: str
    current_iteration: int
    previous_score: float
    critique_history: List[str]
    
    # --- METADATA ---
    mode: str
    topic: str
    run_dir: str
    
    # --- SAFETY / RECOVERY ---
    best_draft_so_far: str
    best_score_so_far: float


def create_initial_state(
    mode: str = "create",
    topic: str = "",
    run_dir: str = ""
) -> PaperState:
    """
    Create a fresh PaperState with default values.
    
    Args:
        mode: "create" or "refine"
        topic: Research topic (for create mode)
        run_dir: Path to this run's output directory
        
    Returns:
        A new PaperState dictionary ready for the pipeline
    """
    return PaperState(
        # Static context - populated by ingestion
        knowledge_base={},
        analysis_context="",
        figure_manifest={},
        evidence_items={},
        
        # Planning
        section_queue=[
            "Introduction",
            "Literature Review", 
            "Methodology",
            "Results",
            "Discussion",
            "Conclusion"
        ],
        current_section="Introduction",
        
        # Iteration memory
        current_draft="",
        current_iteration=0,
        previous_score=0.0,
        critique_history=[],
        
        # Metadata
        mode=mode,
        topic=topic,
        run_dir=run_dir,
        
        # Safety
        best_draft_so_far="",
        best_score_so_far=0.0,
    )
