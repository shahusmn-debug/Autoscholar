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
from .ingestion import ingest_all_references, load_raw_data, load_user_draft, load_raw_context, load_instructions
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
        
        # Load raw context (verbatim, no AI summarization)
        update_display("Ingestion", "Loading raw context...")
        raw_context_dir = f"{inputs_dir}/raw_context"
        raw_context = load_raw_context(raw_context_dir)
        raw_context = load_raw_context(raw_context_dir)
        state["raw_context"] = raw_context
        
        # Load user guidance (instructions.md)
        update_display("Ingestion", "Loading instructions...")
        stats_instr, charts_instr = load_instructions(inputs_dir)
        state["guidance_stats"] = stats_instr
        state["guidance_charts"] = charts_instr
        
        if stats_instr or charts_instr:
            log_event("INFO", "Ingestion", "Loaded user guidance instructions")
        
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
        # Router only decides direction - does NOT modify state
        # State changes happen in update_state_for_next node
        decision = should_continue(state, config)
        return decision
    
    return router


def create_update_state_node(config: dict):
    """
    Create a node that updates state based on review decision.
    
    This node runs BEFORE the conditional edge and prepares state
    for the next iteration or section. When moving sections, it
    saves the best draft achieved for that section.
    """
    def update_state_node(state: PaperState) -> PaperState:
        # Log the decision only here (not in router)
        decision = should_continue(state, config, log_decision=True)
        
        if decision == "retry":
            # Increment iteration and loop back to writer
            state["current_iteration"] = state.get("current_iteration", 0) + 1
            log_event("INFO", "Router", 
                     f"Retrying {state['current_section']}, iteration {state['current_iteration']}")
        
        elif decision in ("next", "end"):
            # Get section_queue first - needed for path construction
            section_queue = state.get("section_queue", [])
            
            # Before moving, ensure we use the best draft for this section
            best_draft = state.get("best_draft_so_far", "")
            current_draft = state.get("current_draft", "")
            best_score = state.get("best_score_so_far", 0.0)
            current_score = state.get("previous_score", 0.0)
            
            # If best draft is better than current, restore it
            if best_draft and best_score > current_score:
                log_event("INFO", "Router", 
                         f"Restoring best draft (score {best_score:.1f} vs current {current_score:.1f})")
            
            # Always save the best draft as the definitive version for this section
            # (even if current == best, to ensure it's available for assembly)
            from pathlib import Path
            run_dir = state.get("run_dir", "runs/default")
            section = state.get("current_section", "")
            section_dir = Path(get_section_dir(run_dir, section))
            section_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine which draft to save as "best"
            final_draft = best_draft if (best_draft and best_score > current_score) else state.get("current_draft", "")
            final_score = best_score if (best_draft and best_score > current_score) else current_score
            
            best_draft_file = section_dir / "best_draft.md"
            with open(best_draft_file, "w", encoding="utf-8") as f:
                f.write(f"<!-- Best Score: {final_score:.1f} -->\n\n{final_draft}")
            
            # Save the completed section's best draft for coherence (for both next AND end)
            current = state.get("current_section", "")
            completed_sections = state.get("completed_sections", {})
            completed_sections[current] = final_draft
            state["completed_sections"] = completed_sections
            
            if decision == "next":
                # Move to next section
                if current in section_queue:
                    idx = section_queue.index(current)
                    if idx + 1 < len(section_queue):
                        state["current_section"] = section_queue[idx + 1]
                        state["current_iteration"] = 0
                        state["current_draft"] = ""
                        state["critique_history"] = []
                        # Reset ALL scoring state for new section
                        state["previous_score"] = 0.0
                        state["best_draft_so_far"] = ""
                        state["best_score_so_far"] = 0.0
                        log_event("INFO", "Router", f"Moving to next section: {state['current_section']}")
        
        return state
    
    return update_state_node


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
    graph.add_node("update_state", create_update_state_node(config))  # New node for state updates
    
    # Set entry point
    graph.set_entry_point("ingestion")
    
    # Add edges for main flow
    graph.add_edge("ingestion", "analyst")
    graph.add_edge("analyst", "writer")
    graph.add_edge("writer", "reviewer")
    graph.add_edge("reviewer", "update_state")  # Always go to update_state first
    
    # Add conditional edges from update_state (after state is updated)
    router = create_router(config)
    graph.add_conditional_edges(
        "update_state",
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
        
        # Run the graph with appropriate recursion limit
        # 6 sections × 4 max iterations × ~2 nodes per iteration = ~50 steps
        update_display("Pipeline", "Starting execution...")
        final_state = graph.invoke(initial_state, config={"recursion_limit": 60})
        
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
    
    def strip_leading_header(content: str) -> str:
        """Strip leading markdown headers (# Section) and HTML comments from draft content."""
        import re
        lines = content.strip().split('\n')
        cleaned_lines = []
        header_stripped = False
        for line in lines:
            # Skip HTML comments like <!-- Best Score: 9.8 -->
            if line.strip().startswith('<!--') and '-->' in line:
                continue
            # Skip first markdown header (# or ##) that matches section name
            if not header_stripped and re.match(r'^#{1,2}\s+', line.strip()):
                header_stripped = True
                continue
            cleaned_lines.append(line)
        return '\n'.join(cleaned_lines).strip()
    
    # Collect best draft from each section
    for section in section_queue:
        section_dir = Path(get_section_dir(run_dir, section))
        
        # First, check for a saved "best" draft (saved when we restore best draft)
        best_draft_file = section_dir / "best_draft.md"
        if best_draft_file.exists():
            content = best_draft_file.read_text(encoding="utf-8")
            content = strip_leading_header(content)
            parts.append(f"\n## {section}\n\n{content}\n")
            continue
        
        # Fall back to the latest iteration
        iter_dirs = sorted(section_dir.glob("iter_*"))
        if iter_dirs:
            latest = iter_dirs[-1]
            draft_file = latest / "draft.md"
            if draft_file.exists():
                content = draft_file.read_text(encoding="utf-8")
                content = strip_leading_header(content)
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

