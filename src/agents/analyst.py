# src/agents/analyst.py
"""
The Analyst Agent (Visualizer)

This agent handles:
1. Reading raw data passports (metadata + samples)
2. Planning needed charts/visualizations
3. Generating Python code to create charts
4. Self-correcting on execution errors
5. Using Vision to interpret generated charts

The Analyst operates in a "Make-then-Look" pattern:
generate code → execute → view result → describe findings
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from ..client import GeminiLLMClient
from ..state import PaperState, FigureItem
from ..tools import execute_python_code, save_code_artifact, save_execution_output
from ..progress import log_event, update_display


# System prompt for chart planning
CHART_PLANNING_PROMPT = """You are a data visualization expert and research analyst.

Given a data passport (column info + sample rows), plan the visualizations needed
to effectively communicate the data's key findings.

For each visualization, specify:
1. Chart type (bar, line, scatter, heatmap, etc.)
2. What variables/columns to use
3. What insight it should reveal
4. A descriptive title

Return JSON:
{
    "charts": [
        {
            "chart_id": "fig_1",
            "chart_type": "line",
            "x_column": "date",
            "y_column": "value",
            "insight": "Show trend over time",
            "title": "Value Trend Over Time"
        }
    ]
}"""


CODE_GENERATION_PROMPT = """You are a Python expert specializing in data visualization.

Generate matplotlib/seaborn code to create the requested chart.
The code will run in a sandboxed environment with these variables pre-defined:
- DATA_DIR: Path to directory containing CSV files
- OUTPUT_DIR: Path to save figures

IMPORTANT RULES:
1. matplotlib is already imported with Agg backend
2. Use pandas to load data: pd.read_csv(f"{DATA_DIR}/filename.csv")
3. Save figures to: plt.savefig(f"{OUTPUT_DIR}/fig_N.png", dpi=150, bbox_inches='tight')
4. Always close figures: plt.close()
5. Handle data cleaning (dropna, type conversion) to avoid errors
6. Use try/except for robustness

Return ONLY the Python code, no markdown formatting."""


SELF_CORRECTION_PROMPT = """You are debugging Python visualization code that failed.

The previous code produced this error:
{error}

Fix the code to handle this error. Common fixes include:
- pd.to_numeric(df['col'], errors='coerce') for type issues
- df.dropna() for null values
- Checking if columns exist before using them
- Handling empty dataframes

Return ONLY the corrected Python code, no markdown."""


VISION_INTERPRETATION_PROMPT = """You are analyzing a data visualization chart.

Describe what this chart shows:
1. What type of chart is this?
2. What are the key trends or patterns?
3. What are the notable data points?
4. What conclusions can be drawn?

Be specific and quantitative where possible."""


def plan_charts(
    client: GeminiLLMClient,
    data_passport: str,
    thinking_level: str = "high"
) -> List[Dict[str, str]]:
    """
    Plan which charts to generate based on the data.
    
    Args:
        client: Gemini client
        data_passport: The data passport text with samples
        thinking_level: Chain-of-thought depth
        
    Returns:
        List of chart specifications
    """
    response_schema = {
        "type": "object",
        "properties": {
            "charts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "chart_id": {"type": "string"},
                        "chart_type": {"type": "string"},
                        "x_column": {"type": "string"},
                        "y_column": {"type": "string"},
                        "insight": {"type": "string"},
                        "title": {"type": "string"}
                    },
                    "required": ["chart_id", "chart_type", "title"]
                }
            }
        },
        "required": ["charts"]
    }
    
    response = client.call_text(
        system_prompt=CHART_PLANNING_PROMPT,
        user_text=f"Plan visualizations for this data:\n\n{data_passport}",
        thinking_level=thinking_level,
        response_schema=response_schema
    )
    
    data = json.loads(response)
    return data.get("charts", [])


def generate_chart_code(
    client: GeminiLLMClient,
    chart_spec: Dict[str, str],
    data_passport: str,
    thinking_level: str = "high"
) -> str:
    """
    Generate Python code to create a specific chart.
    
    Args:
        client: Gemini client
        chart_spec: Chart specification from planning
        data_passport: The data passport for context
        thinking_level: Chain-of-thought depth
        
    Returns:
        Python code as string
    """
    prompt = f"""Create this chart:
- Type: {chart_spec.get('chart_type', 'bar')}
- Title: {chart_spec.get('title', 'Chart')}
- X: {chart_spec.get('x_column', 'N/A')}
- Y: {chart_spec.get('y_column', 'N/A')}
- Goal: {chart_spec.get('insight', 'Visualize the data')}
- Save as: {chart_spec.get('chart_id', 'fig_1')}.png

Data context:
{data_passport[:2000]}"""  # Truncate to avoid context overflow
    
    code = client.call_text(
        system_prompt=CODE_GENERATION_PROMPT,
        user_text=prompt,
        thinking_level=thinking_level
    )
    
    # Clean up markdown code blocks if present
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    
    return code.strip()


def fix_code_error(
    client: GeminiLLMClient,
    original_code: str,
    error: str,
    thinking_level: str = "high"
) -> str:
    """
    Attempt to fix code that produced an error.
    
    Args:
        client: Gemini client
        original_code: The code that failed
        error: The error message
        thinking_level: Chain-of-thought depth
        
    Returns:
        Corrected Python code
    """
    prompt = SELF_CORRECTION_PROMPT.format(error=error)
    prompt += f"\n\nOriginal code:\n```python\n{original_code}\n```"
    
    code = client.call_text(
        system_prompt=prompt,
        user_text="Fix this code to handle the error.",
        thinking_level=thinking_level
    )
    
    # Clean up markdown
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    
    return code.strip()


def interpret_chart(
    client: GeminiLLMClient,
    image_path: str,
    chart_title: str,
    thinking_level: str = "high"
) -> str:
    """
    Use Vision to interpret a generated chart.
    
    Args:
        client: Gemini client
        image_path: Path to the chart image
        chart_title: Title for context
        thinking_level: Chain-of-thought depth
        
    Returns:
        Interpretation text
    """
    response = client.call_images(
        system_prompt=VISION_INTERPRETATION_PROMPT,
        user_text=f"Analyze this chart titled: '{chart_title}'",
        image_paths=[image_path],
        thinking_level=thinking_level
    )
    
    return response


def node_analyst(state: PaperState, client: GeminiLLMClient, config: dict) -> PaperState:
    """
    The Analyst node for the LangGraph pipeline.
    
    This node:
    1. Reads data passport from raw_data
    2. Plans necessary visualizations
    3. Generates and executes code (with self-correction)
    4. Interprets results with Vision
    5. Updates state with analysis_context and figure_manifest
    
    Args:
        state: Current pipeline state
        client: Gemini client
        config: Configuration dictionary
        
    Returns:
        Updated state
    """
    run_dir = state["run_dir"]
    figures_dir = str(Path(run_dir) / "figures")
    tools_dir = str(Path(run_dir) / "tools")
    transcripts_dir = str(Path(run_dir) / "transcripts")
    
    # Get thinking level from config
    thinking_level = config.get("generation", {}).get("analyst", {}).get("thinking_level", "high")
    max_retries = 3
    
    update_display("Analyst", "Reading data passport...")
    log_event("INFO", "Analyst", "Starting analysis phase")
    
    # Get data passport from evidence items or load fresh
    if state.get("evidence_items"):
        # Build passport from evidence
        passport_parts = []
        for ev_id, item in state["evidence_items"].items():
            passport_parts.append(f"=== {ev_id}: {item.source_file} ===\n{item.data_snippet}\n")
        data_passport = "\n".join(passport_parts)
    else:
        # Try loading from inputs
        from ..ingestion import load_raw_data
        evidence_items, data_passport = load_raw_data("inputs/raw_data")
        state["evidence_items"] = evidence_items
    
    # If no data, skip chart generation
    if not data_passport or data_passport == "No CSV data files found.":
        log_event("INFO", "Analyst", "No CSV data found, skipping chart generation")
        state["analysis_context"] = "No raw data available for analysis."
        state["figure_manifest"] = {}
        return state
    
    # Plan charts
    update_display("Analyst", "Planning visualizations...")
    charts = plan_charts(client, data_passport, thinking_level)
    log_event("INFO", "Analyst", f"Planned {len(charts)} charts")
    
    figure_manifest = {}
    interpretations = []
    
    for chart in charts:
        chart_id = chart.get("chart_id", f"fig_{len(figure_manifest)+1}")
        update_display("Analyst", f"Generating {chart_id}...")
        log_event("INFO", "Analyst", f"Generating code for {chart_id}")
        
        # Generate code
        code = generate_chart_code(client, chart, data_passport, thinking_level)
        attempt = 1
        success = False
        
        # Self-correction loop
        while attempt <= max_retries and not success:
            # Save code artifact
            code_path = save_code_artifact(code, chart_id, attempt, tools_dir)
            log_event("INFO", "Analyst", f"Executing {chart_id} (attempt {attempt})")
            
            # Execute
            result = execute_python_code(code, figures_dir, "inputs/raw_data")
            save_execution_output(result, chart_id, attempt, tools_dir)
            
            if result.success:
                success = True
                # Find the generated figure
                fig_path = None
                for f in result.output_files:
                    if f.endswith('.png'):
                        fig_path = f
                        break
                
                if fig_path:
                    log_event("INFO", "Analyst", f"Successfully generated {chart_id}")
                    
                    # Interpret with Vision
                    update_display("Analyst", f"Analyzing {chart_id} with Vision...")
                    interpretation = interpret_chart(client, fig_path, chart.get("title", chart_id))
                    
                    figure_manifest[chart_id] = FigureItem(
                        figure_id=chart_id,
                        file_path=fig_path,
                        code_path=code_path,
                        interpretation=interpretation,
                        suggested_caption=chart.get("title", f"Figure: {chart_id}"),
                        data_source=chart.get("x_column", "raw_data")
                    )
                    
                    interpretations.append(f"### {chart.get('title', chart_id)}\n{interpretation}")
                else:
                    log_event("WARNING", "Analyst", f"Code ran but no PNG created for {chart_id}")
            else:
                # Self-correction
                log_event("WARNING", "Analyst", f"Attempt {attempt} failed: {result.error[:100]}")
                
                if attempt < max_retries:
                    update_display("Analyst", f"Fixing {chart_id} (attempt {attempt+1})...")
                    code = fix_code_error(client, code, result.error, thinking_level)
                
                attempt += 1
        
        if not success:
            log_event("ERROR", "Analyst", f"Failed to generate {chart_id} after {max_retries} attempts")
    
    # Compile analysis context
    analysis_context = "# Data Analysis Report\n\n"
    analysis_context += f"Generated {len(figure_manifest)} visualizations.\n\n"
    analysis_context += "\n\n".join(interpretations)
    
    state["analysis_context"] = analysis_context
    state["figure_manifest"] = figure_manifest
    
    log_event("INFO", "Analyst", f"Completed analysis with {len(figure_manifest)} figures")
    
    return state
