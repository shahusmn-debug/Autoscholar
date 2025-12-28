# src/agents/writer.py
"""
The Writer Agent (Author/Editor Dual Persona)

This agent has two modes:
1. Author Mode (create mode or iteration > 1): Writes sections from scratch
2. Editor Mode (refine mode, iteration 1): Preserves voice while fixing issues

Key responsibilities:
- Write publication-quality academic prose
- Cite references using knowledge_base keys
- Incorporate analysis findings
- Address reviewer feedback
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

from ..client import GeminiLLMClient
from ..state import PaperState
from ..progress import log_event, update_display


# Author persona for creating content
AUTHOR_SYSTEM_PROMPT = """You are an expert academic author writing a section of a research paper.

ROLE: Write clear, precise, publication-quality prose.

INPUTS:
- Section: The section you are writing (e.g., Introduction, Methodology)
- Topic: The research topic
- Knowledge Base: Extracted summaries from reference papers
- Analysis Context: Findings from data analysis with figures
- Critique History: Feedback from previous iterations (if any)

REQUIREMENTS:
1. Use formal academic tone appropriate for the section
2. Cite references as [Ref_XXX] matching knowledge_base keys
3. Base factual claims on analysis_context (data ground truth)
4. Address all points in critique_history if provided
5. Structure with clear paragraphs and logical flow

For each section type:
- Introduction: Context, problem statement, research questions, contribution summary
- Literature Review: Synthesize prior work, identify gaps
- Methodology: Detailed procedures, justify choices
- Results: Present findings, reference figures
- Discussion: Interpret results, compare to literature, limitations
- Conclusion: Summarize contributions, future directions

OUTPUT: Well-structured markdown for the section."""


# Editor persona for refinement
EDITOR_SYSTEM_PROMPT = """You are an expert Academic Editor refining a draft paper.

ROLE: Improve the draft while preserving the original author's voice.

INPUTS:
- Draft: The existing text to refine
- Knowledge Base: Reference papers with citation keys
- Analysis Context: Data analysis findings (GROUND TRUTH)
- Critique: Specific issues to address

CRITICAL RULES:
1. PRESERVE the author's voice and style where possible
2. NORMALIZE citations to match knowledge_base keys:
   - Change "(Smith 2023)" → "[Ref_001]" (find matching reference)
3. CORRECT factual claims that contradict analysis_context
   - The data analysis is ground truth; the text must match it
4. Address all critique points systematically
5. Improve clarity and flow without rewriting unnecessarily

OUTPUT: The refined section in markdown, with changes tracked in your explanation."""


def build_knowledge_context(knowledge_base: dict) -> str:
    """Build a text summary of the knowledge base for the prompt."""
    if not knowledge_base:
        return "No references available."
    
    parts = ["## Available References\n"]
    for ref_id, item in knowledge_base.items():
        title = getattr(item, 'title', None) or "Untitled"
        authors = getattr(item, 'authors', None) or "Unknown"
        year = getattr(item, 'year', None) or ""
        summary = getattr(item, 'summary', '') or ""
        
        parts.append(f"### {ref_id}: {title}")
        if authors or year:
            parts.append(f"*{authors} {year}*")
        if summary:
            parts.append(summary[:500] + "..." if len(summary) > 500 else summary)
        parts.append("")
    
    return "\n".join(parts)


def build_figure_context(figure_manifest: dict) -> str:
    """Build a text summary of available figures."""
    if not figure_manifest:
        return "No figures available."
    
    parts = ["## Available Figures\n"]
    for fig_id, item in figure_manifest.items():
        caption = getattr(item, 'suggested_caption', fig_id)
        interpretation = getattr(item, 'interpretation', '')
        
        parts.append(f"### {fig_id}: {caption}")
        if interpretation:
            parts.append(interpretation[:300] + "..." if len(interpretation) > 300 else interpretation)
        parts.append("")
    
    return "\n".join(parts)


def node_writer(state: PaperState, client: GeminiLLMClient, config: dict) -> PaperState:
    """
    The Writer node for the LangGraph pipeline.
    
    Switches between Author and Editor personas based on:
    - mode == "refine" AND iteration == 1 → Editor
    - Otherwise → Author
    
    Args:
        state: Current pipeline state
        client: Gemini client
        config: Configuration dictionary
        
    Returns:
        Updated state with new draft
    """
    mode = state.get("mode", "create")
    iteration = state.get("current_iteration", 0)
    section = state.get("current_section", "Introduction")
    
    thinking_level = config.get("generation", {}).get("writer", {}).get("thinking_level", "high")
    
    # Determine persona
    use_editor = (mode == "refine" and iteration == 1)
    persona = "Editor" if use_editor else "Author"
    
    update_display("Writer", f"[{persona}] Writing {section}...")
    log_event("INFO", "Writer", f"Starting {section} with {persona} persona (iter {iteration})")
    
    # Build context strings
    knowledge_context = build_knowledge_context(state.get("knowledge_base", {}))
    analysis_context = state.get("analysis_context", "No analysis available.")
    figure_context = build_figure_context(state.get("figure_manifest", {}))
    
    # Build critique history string
    critique_history = state.get("critique_history", [])
    if critique_history:
        critique_text = "\n\n---\n".join(critique_history)
    else:
        critique_text = "No previous feedback."
    
    # Select system prompt
    system_prompt = EDITOR_SYSTEM_PROMPT if use_editor else AUTHOR_SYSTEM_PROMPT
    
    # Build user prompt based on persona
    if use_editor:
        current_draft = state.get("current_draft", "")
        user_prompt = f"""## Section: {section}

## Original Draft to Refine:
{current_draft}

## Knowledge Base:
{knowledge_context}

## Data Analysis (GROUND TRUTH):
{analysis_context}

{figure_context}

## Issues to Address:
{critique_text}

Please refine this draft, preserving the author's voice while fixing issues."""

    else:
        topic = state.get("topic", "Research Topic")
        user_prompt = f"""## Section to Write: {section}

## Research Topic: {topic}

## Knowledge Base:
{knowledge_context}

## Data Analysis:
{analysis_context}

{figure_context}

## Feedback to Address:
{critique_text}

Write the {section} section for this research paper."""

    # Generate the draft
    new_draft = client.call_text(
        system_prompt=system_prompt,
        user_text=user_prompt,
        thinking_level=thinking_level
    )
    
    # Update state
    state["current_draft"] = new_draft
    
    # Track best draft
    current_score = state.get("previous_score", 0.0)
    if current_score > state.get("best_score_so_far", 0.0):
        state["best_draft_so_far"] = new_draft
        state["best_score_so_far"] = current_score
    
    log_event("INFO", "Writer", f"Completed {section} draft ({len(new_draft.split())} words)")
    
    # Save draft to disk
    from ..utils import save_iteration_draft
    run_dir = state.get("run_dir", "runs/default")
    save_iteration_draft(run_dir, section, iteration, new_draft)
    
    return state
