# src/agents/reviewer.py
"""
The Reviewer Agent

This agent:
1. Scores drafts on a 0.0-10.0 scale
2. Compares claims against knowledge_base and analysis_context
3. Provides specific, actionable feedback
4. Enables the iteration loop by setting scores

The Reviewer is intentionally critical but fair.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple

from ..client import GeminiLLMClient
from ..state import PaperState
from ..progress import log_event, update_display


REVIEWER_SYSTEM_PROMPT = """You are a critical but fair peer reviewer for academic papers.

ROLE: Evaluate the draft section against provided evidence and standards.

SCORING CRITERIA (0.0 - 10.0):
- 9.5-10.0: Publication ready, no issues
- 8.0-9.4: Minor improvements needed
- 6.0-7.9: Significant revision required
- 4.0-5.9: Major problems, substantial rewrite needed
- 0.0-3.9: Fundamental issues, not acceptable

EVALUATION CHECKLIST:
1. EVIDENCE SUPPORT: Are the claims made supported by the analysis_context?
   - Note: The paper does NOT need to include all available data, only what is relevant.
2. CITATIONS: Are references properly cited with [Ref_XXX] format?
3. CLARITY: Is the writing clear, well-organized, and compelling?
4. COMPLETENESS: Is the section thorough for its type?
5. ACADEMIC TONE: Is the tone appropriate for publication?
6. SELECTIVITY: Did the author choose the most relevant evidence to tell a coherent story?

GROUND TRUTH RULE: Claims must NOT contradict the analysis_context. However, interpretation of the data and omission of irrelevant details is encouraged.

EVIDENCE-ONLY RULE: Do NOT assume or infer details that are not EXPLICITLY stated in the 
provided evidence.

FEEDBACK REQUIREMENTS:
- Be specific about issues (cite line/paragraph)
- Explain why something is wrong
- Suggest how to fix it
- Prioritize issues by severity

Return JSON:
{
    "score": 7.5,
    "feedback": "Detailed feedback text..."
}"""


def node_reviewer(state: PaperState, client: GeminiLLMClient, config: dict) -> PaperState:
    """
    The Reviewer node for the LangGraph pipeline.
    
    Evaluates the current draft against:
    - Knowledge base (citation accuracy)
    - Analysis context (factual accuracy)
    - Academic writing standards
    
    Args:
        state: Current pipeline state
        client: Gemini client
        config: Configuration dictionary
        
    Returns:
        Updated state with score and critique
    """
    section = state.get("current_section", "Introduction")
    iteration = state.get("current_iteration", 0)
    draft = state.get("current_draft", "")
    mode = state.get("mode", "create")
    
    thinking_level = config.get("generation", {}).get("reviewer", {}).get("thinking_level", "low")
    
    update_display("Reviewer", f"Evaluating {section} (iter {iteration})...")
    log_event("INFO", "Reviewer", f"Starting review of {section}, iteration {iteration}")
    
    # Build context
    analysis_context = state.get("analysis_context", "No analysis available.")
    raw_context_summary = state.get("raw_context_summary", "No raw context summary available.")
    
    # Build knowledge base summary
    kb_parts = []
    for ref_id, item in state.get("knowledge_base", {}).items():
        title = getattr(item, 'title', None) or "Untitled"
        kb_parts.append(f"- {ref_id}: {title}")
    knowledge_summary = "\n".join(kb_parts) if kb_parts else "No references available."
    
    # Response schema for structured output
    response_schema = {
        "type": "object",
        "properties": {
            "score": {"type": "number"},
            "feedback": {"type": "string"}
        },
        "required": ["score", "feedback"]
    }
    
    # Build review prompt
    context_note = ""
    if mode == "refine" and iteration == 0:
        context_note = "\n\n**NOTE**: This is the user's original draft (iteration 0). " \
                      "Evaluate it to identify improvements needed."
    
    user_prompt = f"""## Section: {section}
## Iteration: {iteration}
{context_note}

## Draft to Review:
{draft}

## Data Analysis (GROUND TRUTH):
{analysis_context}

## Methodology Context (from code/specs):
{raw_context_summary}

## Available References:
{knowledge_summary}

Please evaluate this draft and provide a score (0.0-10.0) with detailed feedback."""

    try:
        response = client.call_text(
            system_prompt=REVIEWER_SYSTEM_PROMPT,
            user_text=user_prompt,
            thinking_level=thinking_level,
            response_schema=response_schema
        )
        
        data = json.loads(response)
        score = float(data.get("score", 0.0))
        feedback = data.get("feedback", "No feedback provided.")
        
    except Exception as e:
        log_event("ERROR", "Reviewer", f"Review failed: {str(e)}")
        score = 0.0
        feedback = f"Review failed due to error: {str(e)}"
    
    # Log the review result
    log_event("INFO", "Reviewer", f"Iteration {iteration} Score: {score:.1f}")
    
    # Handle score reporting differently based on context
    if mode == "refine" and iteration == 0:
        log_event("INFO", "Reviewer", f"User draft score: {score:.1f}. Issues: {feedback[:100]}...")
    else:
        log_event("INFO", "Reviewer", f"Draft score: {score:.1f}")
    
    # Update state
    state["previous_score"] = score
    state["critique_history"].append(f"### Iteration {iteration} (Score: {score:.1f})\n{feedback}")
    
    # Track best draft
    if score > state.get("best_score_so_far", 0.0):
        state["best_score_so_far"] = score
        state["best_draft_so_far"] = draft
        
        # Eagerly save validation/best draft to disk to prevent data loss
        # especially for the final section where update_state might behave differently
        from ..utils import get_section_dir
        run_dir = state.get("run_dir", "runs/default")
        section_dir = Path(get_section_dir(run_dir, section))
        best_draft_file = section_dir / "best_draft.md"
        try:
            with open(best_draft_file, "w", encoding="utf-8") as f:
                f.write(f"<!-- Best Score: {score:.1f} -->\n\n{draft}")
            log_event("INFO", "Reviewer", f"New best score ({score:.1f})! Saved best_draft.md")
        except Exception as e:
            log_event("WARNING", "Reviewer", f"Failed to save best_draft.md: {e}")
    
    # Save critique to disk
    from ..utils import save_iteration_draft
    run_dir = state.get("run_dir", "runs/default")
    save_iteration_draft(run_dir, section, iteration, draft, feedback)
    
    return state


def should_continue(state: PaperState, config: dict, log_decision: bool = False) -> str:
    """
    Router logic to decide next action after review.
    
    Returns:
    - "retry": Continue iterating on current section
    - "next": Move to next section
    - "end": All sections complete
    
    Decision Logic (simple):
    1. Score >= threshold AND iteration >= min → next section (quality met)
    2. Iteration >= max → next section (use best draft)
    3. Otherwise → retry (keep iterating)
    
    Args:
        state: Current pipeline state
        config: Configuration dict
        log_decision: If True, log the decision reason. Only set True from one call site.
    """
    score = state.get("previous_score", 0.0)
    iteration = state.get("current_iteration", 0)
    section_queue = state.get("section_queue", [])
    current_section = state.get("current_section", "")
    
    limits = config.get("iteration_limits", {})
    min_iter = limits.get("min", 1)
    max_iter = limits.get("max", 4)
    threshold = limits.get("score_threshold", 9.0)
    
    def move_to_next_or_end():
        """Helper to determine if we go to next section or end."""
        if current_section in section_queue:
            idx = section_queue.index(current_section)
            if idx + 1 < len(section_queue):
                return "next"
        return "end"
    
    # 1. Check if we've hit the quality threshold
    # Note: iteration is 0-indexed, so iteration 0 = "completed 1 iteration"
    completed_iterations = iteration + 1
    if score >= threshold and completed_iterations >= min_iter:
        if log_decision:
            log_event("INFO", "Reviewer", f"Quality threshold {threshold} met for {current_section}")
        return move_to_next_or_end()
    
    # 2. Hard maximum - stop and use best draft
    if completed_iterations >= max_iter:
        if log_decision:
            log_event("WARNING", "Reviewer", f"Hit max iterations ({max_iter}) for {current_section}. Using best draft.")
        return move_to_next_or_end()
    
    # 3. Keep iterating
    return "retry"

