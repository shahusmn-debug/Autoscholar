# src/graph.py
"""
LangGraph State Machine - Pipeline Orchestration

This module defines the cyclic state graph that orchestrates
the research paper writing pipeline:

    Ingestion → Analyst → Writer → Reviewer → (loop or next section)

The graph handles:
- Initial reference and data ingestion
- Cyclic write/review loops per section
- Section progression
- Final paper assembly
"""

from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END

from .state import PaperState, create_initial_state
from .client import GeminiLLMClient
from .ingestion import ingest_all_references, load_raw_data, load_user_draft
from .agents.analyst import node_analyst
from .agents.writer import node_writer
from .agents.reviewer import node_reviewer, should_continue
from .utils import (
    create_run_directory, save_knowledge_base, save_final_paper,
    get_section_dir
)
from .progress import log_event, update_display, start_progress, stop_progress


def create_ingestion_node(client: GeminiLLMClient, config: dict, inputs_dir: str):
    """
    Factory function to create the ingestion node.
    
    This node loads all input data into the state:
    - References (PDFs and text files)
    - Raw data (CSVs)
    - User draft (for refine mode)
    """
    def node_ingestion(state: PaperState) -> PaperState:
        run_dir = state["run_dir"]
        mode = state.get("mode", "create")
        transcripts_dir = f"{run_dir}/transcripts"
        
        thinking_level = config.get("generation", {}).get("ingestion", {}).get("thinking_level", "high")
        
        update_display("Ingestion", "Loading references...")
        log_event("INFO", "Ingestion", f"Starting ingestion in {mode} mode")
        
        # Load references
        references_dir = f"{inputs_dir}/references"
        knowledge_base = ingest_all_references(
            references_dir, client, transcripts_dir, thinking_level
        )
        state["knowledge_base"] = knowledge_base
        
        # Load raw data
        update_display("Ingestion", "Loading raw data...")
        raw_data_dir = f"{inputs_dir}/raw_data"
        evidence_items, data_passport = load_raw_data(raw_data_dir)
        state["evidence_items"] = evidence_items
        
        # For refine mode, load the user draft
        if mode == "refine":
            update_display("Ingestion", "Loading user draft...")
            draft_dir = f"{inputs_dir}/user_draft"
            user_draft = load_user_draft(draft_dir)
            if user_draft:
                state["current_draft"] = user_draft
                word_count = len(user_draft.split())
                log_event("INFO", "Ingestion", 
                         f"Refinement Mode Active. Loaded user draft ({word_count} words)")
            else:
                log_event("WARNING", "Ingestion", 
                         "Refine mode but no draft.md found. Switching to create mode.")
                state["mode"] = "create"
        
        # Save knowledge base to disk
        save_knowledge_base(run_dir, knowledge_base, data_passport)
        
        log_event("INFO", "Ingestion", 
                 f"Ingestion complete: {len(knowledge_base)} refs, {len(evidence_items)} data files")
        
        return state
    
    return node_ingestion


def create_analyst_node(client: GeminiLLMClient, config: dict):
    """Factory to create analyst node with client and config bound."""
    def wrapped_analyst(state: PaperState) -> PaperState:
        return node_analyst(state, client, config)
    return wrapped_analyst


def create_writer_node(client: GeminiLLMClient, config: dict):
    """Factory to create writer node with client and config bound."""
    def wrapped_writer(state: PaperState) -> PaperState:
        return node_writer(state, client, config)
    return wrapped_writer


def create_reviewer_node(client: GeminiLLMClient, config: dict):
    """Factory to create reviewer node with client and config bound."""
    def wrapped_reviewer(state: PaperState) -> PaperState:
        return node_reviewer(state, client, config)
    return wrapped_reviewer


def create_router(config: dict):
    """Create the routing function for post-review decisions."""
    def router(state: PaperState) -> Literal["retry", "next", "end"]:
        decision = should_continue(state, config)
        
        if decision == "retry":
            # Increment iteration and loop back to writer
            state["current_iteration"] = state.get("current_iteration", 0) + 1
            log_event("INFO", "Router", 
                     f"Retrying {state['current_section']}, iteration {state['current_iteration']}")
        
        elif decision == "next":
            # Move to next section
            section_queue = state.get("section_queue", [])
            current = state.get("current_section", "")
            if current in section_queue:
                idx = section_queue.index(current)
                if idx + 1 < len(section_queue):
                    state["current_section"] = section_queue[idx + 1]
                    state["current_iteration"] = 0
                    state["current_draft"] = ""
                    state["critique_history"] = []
                    log_event("INFO", "Router", f"Moving to next section: {state['current_section']}")
        
        return decision
    
    return router


def build_graph(client: GeminiLLMClient, config: dict, inputs_dir: str) -> StateGraph:
    """
    Build the LangGraph state machine.
    
    Graph structure:
    
    START → ingestion → analyst → writer → reviewer → router
                                    ↑                    ↓
                                    └─────── retry ──────┘
                                                         ↓ (next/end)
                                                       END
    
    Args:
        client: Gemini client
        config: Configuration dictionary
        inputs_dir: Path to inputs directory
        
    Returns:
        Compiled StateGraph
    """
    # Create the graph with PaperState as the state type
    graph = StateGraph(PaperState)
    
    # Create nodes with bound dependencies
    graph.add_node("ingestion", create_ingestion_node(client, config, inputs_dir))
    graph.add_node("analyst", create_analyst_node(client, config))
    graph.add_node("writer", create_writer_node(client, config))
    graph.add_node("reviewer", create_reviewer_node(client, config))
    
    # Set entry point
    graph.set_entry_point("ingestion")
    
    # Add edges for main flow
    graph.add_edge("ingestion", "analyst")
    graph.add_edge("analyst", "writer")
    graph.add_edge("writer", "reviewer")
    
    # Add conditional edges from reviewer
    router = create_router(config)
    graph.add_conditional_edges(
        "reviewer",
        router,
        {
            "retry": "writer",
            "next": "writer",  # Writer will pick up new section from state
            "end": END
        }
    )
    
    return graph.compile()


def run_pipeline(
    mode: str = "create",
    topic: str = "",
    config: dict = None,
    inputs_dir: str = "inputs",
    runs_dir: str = "runs"
) -> str:
    """
    Execute the full research paper pipeline.
    
    Args:
        mode: "create" or "refine"
        topic: Research topic (for create mode)
        config: Configuration dictionary
        inputs_dir: Path to inputs directory
        runs_dir: Base path for run outputs
        
    Returns:
        Path to the final paper
    """
    if config is None:
        from .utils import load_config
        config = load_config()
    
    # Create run directory
    run_dir = create_run_directory(runs_dir)
    
    # Start progress tracking
    start_progress(run_dir)
    
    log_event("INFO", "Setup", f"Starting pipeline in {mode} mode")
    if topic:
        log_event("INFO", "Setup", f"Topic: {topic}")
    
    try:
        # Initialize client
        models = config.get("models", {})
        api_config = config.get("google_genai", {})
        
        client = GeminiLLMClient(
            model_text=models.get("text", "gemini-2.5-pro-preview-06-05"),
            model_vision=models.get("vision", "gemini-2.5-pro-preview-06-05"),
            api_version_text=api_config.get("api_version_text", "v1beta"),
            api_version_vision=api_config.get("api_version_vision", "v1alpha")
        )
        
        # Build graph
        graph = build_graph(client, config, inputs_dir)
        
        # Create initial state
        initial_state = create_initial_state(mode=mode, topic=topic, run_dir=run_dir)
        
        # Run the graph
        update_display("Pipeline", "Starting execution...")
        final_state = graph.invoke(initial_state)
        
        # Assemble final paper
        update_display("Pipeline", "Assembling final paper...")
        final_content = assemble_final_paper(final_state, run_dir)
        final_path = save_final_paper(run_dir, final_content)
        
        log_event("INFO", "Pipeline", f"Complete! Final paper: {final_path}")
        
        return final_path
        
    except Exception as e:
        log_event("ERROR", "Pipeline", f"Pipeline failed: {str(e)}")
        raise
    finally:
        stop_progress()


def assemble_final_paper(state: PaperState, run_dir: str) -> str:
    """
    Assemble all sections into the final paper.
    
    Args:
        state: Final pipeline state
        run_dir: Run directory path
        
    Returns:
        Complete paper as markdown string
    """
    from pathlib import Path
    
    section_queue = state.get("section_queue", [])
    parts = ["# Research Paper\n"]
    
    if state.get("topic"):
        parts.append(f"**Topic:** {state['topic']}\n")
    
    parts.append("---\n")
    
    # Collect best draft from each section
    for section in section_queue:
        section_dir = Path(get_section_dir(run_dir, section))
        
        # Find the latest iteration
        iter_dirs = sorted(section_dir.glob("iter_*"))
        if iter_dirs:
            latest = iter_dirs[-1]
            draft_file = latest / "draft.md"
            if draft_file.exists():
                content = draft_file.read_text(encoding="utf-8")
                parts.append(f"\n## {section}\n\n{content}\n")
    
    # If no sections found, use best draft from state
    if len(parts) <= 3 and state.get("best_draft_so_far"):
        parts.append(f"\n{state['best_draft_so_far']}\n")
    
    # Add figure references
    if state.get("figure_manifest"):
        parts.append("\n---\n\n## Figures\n")
        for fig_id, item in state["figure_manifest"].items():
            caption = getattr(item, 'suggested_caption', fig_id)
            path = getattr(item, 'file_path', '')
            parts.append(f"\n### {fig_id}: {caption}\n")
            parts.append(f"![{caption}]({path})\n")
    
    return "\n".join(parts)
