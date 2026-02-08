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
import time
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

Generate matplotlib code to create the requested chart.
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
6. Use try/except for robustness
7. ALLOWED LIBRARIES: pandas, numpy, matplotlib, seaborn, scipy, statsmodels
8. DO NOT use any other libraries (no plotly, no sklearn, etc)

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


BATCH_VISION_PROMPT = """You are analyzing multiple data visualization charts from a research study.

For EACH figure provided, give a brief interpretation covering:
1. Chart type and what it shows
2. Key pattern or finding
3. Notable values

Format your response as:
### [Figure filename]
[Your interpretation]

### [Next figure filename]
[Your interpretation]

Be concise but specific. Reference actual values from the charts where visible."""


RAW_CONTEXT_ANALYSIS_PROMPT = """You are a research methodology expert analyzing code and documentation files.

Your task is to create a comprehensive summary of the provided raw context (code, scripts, documentation) that will be used to:
1. Help writers accurately describe the methodology
2. Help reviewers verify claims match the actual implementation
3. Provide grounding for the entire research paper

ANALYSIS REQUIREMENTS:
1. **Data Collection**: What data sources are used? How is data collected/accessed?
2. **Data Processing**: What cleaning, filtering, or transformation steps are applied?
3. **Analysis Methods**: What statistical or analytical techniques are used?
4. **Key Variables**: What are the main variables, features, or columns analyzed?
5. **Technical Details**: Any specific libraries, APIs, or tools used?
6. **Limitations**: Any apparent limitations in the methodology?

OUTPUT FORMAT:
Provide a detailed but organized summary that researchers can reference.
Use clear sections and bullet points for easy scanning.
Include specific function names, variable names, and technical details where helpful."""


# New: Stats computation before chart generation
STATS_COMPUTATION_PROMPT = """You are a data analyst. Given the data structure below, generate Python code to compute KEY STATISTICS that would be needed for a research paper.

The code will run in a sandboxed environment with these PYTHON VARIABLES already defined:
- DATA_DIR: String variable containing the absolute path to CSV files
- OUTPUT_DIR: String variable containing the absolute path to save output

CRITICAL: DATA_DIR and OUTPUT_DIR are ALREADY DEFINED as Python string variables.
Do NOT use os.environ.get() - just use them directly like: pd.read_csv(f"{DATA_DIR}/file.csv")

REQUIREMENTS:
1. **Inspect and Clean Data**: Before stats, inspect columns (strip whitespace!), handle numeric conversions, and drop empty rows.
2. Compute groupby aggregations for categorical vs numeric columns (mean, median, count, std)
3. Compute overall distributions and key percentiles
4. If there are obvious comparison groups (e.g., categories), compute stats for each group
5. PERFORM STATISTICAL TESTS (Inferential Statistics):
   - If 2 groups + numeric data: Run T-test (scipy.stats.ttest_ind)
   - If >2 groups + numeric data: Run ANOVA (scipy.stats.f_oneway)
   - If categorical vs categorical: Run Chi-square test
   - Report p-values and significance levels (e.g., "p < 0.05")
6. INCLUDE 95% CONFIDENCE INTERVALS where appropriate:
   - For group means: report as "mean [CI_lower, CI_upper]"
   - Use scipy.stats.t.interval() or compute as mean ± (t_critical * sem)
   - CIs are especially important for group comparisons
7. **Consider THRESHOLDS**: Check if values below a certain threshold (in X) correspond to ZERO or negligible risk (in Y).
8. Save results to f"{OUTPUT_DIR}/computed_stats.md" as a readable markdown file
9. START the output with a "## Methodology Summary" section that describes:
   - What data files were loaded
   - What key variables/columns were analyzed
   - What statistical methods were applied (be specific: mean, t-test, etc.)
   - This section helps downstream agents understand how results were derived

=== METHODOLOGY CONTEXT (FROM RAW CODE/DOCS) ===
{methodology_summary}

=== USER GUIDANCE (MUST FOLLOW) ===
{user_guidance}

IMPORTANT RULES:
1. Use DATA_DIR directly: pd.read_csv(f"{DATA_DIR}/filename.csv")
2. Use OUTPUT_DIR directly: open(f"{OUTPUT_DIR}/computed_stats.md", "w")
3. Output should be a MARKDOWN file with clear section headers
4. Include the actual numeric values (e.g., "blue_collar_mean: 0.028")
5. Handle missing values gracefully
6. Use try/except for robustness
7. ALLOWED LIBRARIES: pandas, numpy, scipy, statsmodels, tabulate
8. DO NOT use any other libraries

Return ONLY executable Python code, no markdown formatting."""


# New: Batched chart generation with ground truth
BATCHED_CHART_PROMPT = """You are a data visualization expert. Generate ALL charts needed to visualize this data.

CRITICAL: You have been provided with COMPUTED STATISTICS (ground truth). 
Your charts MUST accurately visualize these pre-computed values.

The code will run in a sandboxed environment with these PYTHON VARIABLES already defined:
- DATA_DIR: String variable containing the absolute path to CSV files
- OUTPUT_DIR: String variable containing the absolute path to save figures

IMPORTANT: DATA_DIR and OUTPUT_DIR are ALREADY DEFINED Python string variables.
Do NOT use os.environ.get() or define them yourself. Just use them directly.

RECOMMENDED VISUALIZATION TYPES (Select those applicable to this data):
1. **Univariate Analysis**: Visualize distributions of key numeric variables (e.g., histograms, boxplots).
2. **Group Comparisons**: If categorical groups exist, comparisons with error bars (CI 95%) are highly valued.
3. **Trends**: If temporal data exists (dates/timestamps), visualize trends over time.
4. **Relationships**: If multiple numeric variables exist, show correlations (scatter, heatmaps).
5. **Composition/Volume**: Visualize counts or proportions of top categories/groups.

=== METHODOLOGY CONTEXT (FROM RAW CODE/DOCS) ===
{methodology_summary}

=== USER GUIDANCE (MUST FOLLOW) ===
{user_guidance}

MINIMUM REQUIREMENTS:
- Generate a minimum of 4 distinct charts that best tell the data's story.
- Prioritize charts that support the statistical findings.
- Ensure all charts are "publication ready" (clear titles, labels, legends).

QUALITY REQUIREMENTS:
- **Axis Ranges**: Set appropriate axis limits that show the data clearly. If data has natural bounds (e.g., sentiment -1 to 1), use them.
- **Error Bars**: Include 95% confidence intervals on bar charts comparing groups.
- **Consistent Palette**: Use a colorblind-friendly palette. Suggestion: '#1f77b4' for group1, '#ff7f0e' for group2.
- **Labels**: After any sort operation, reset index or use enumerate() for correct label positioning.
- **Annotations**: Annotate key values (means, counts) directly on charts where helpful.

Return JSON with this structure:
{
    "charts": [
        {
            "chart_id": "fig_descriptive_name",
            "title": "Human-readable title",
            "chart_type": "bar|line|box|scatter|histogram|lollipop",
            "insight": "What this chart reveals",
            "ground_truth_values": {"key": "value pairs from computed stats"},
            "code": "complete python code string"
        }
    ]
}

RULES FOR EACH CHART:
1. matplotlib is already imported with Agg backend
2. Use pd.read_csv(f"{DATA_DIR}/filename.csv")
3. Save as plt.savefig(f"{OUTPUT_DIR}/{chart_id}.png", dpi=300, bbox_inches='tight')
4. Always plt.close() at the end
5. Handle data cleaning (dropna, type conversion)
6. Use try/except for robustness
7. ALLOWED LIBRARIES: pandas, numpy, matplotlib, seaborn, scipy, statsmodels
8. DO NOT use any other libraries (no plotly, no sklearn, etc)
9. The chart MUST match the ground_truth_values you specify
10. After sorting DataFrames, reset index before using positional labels"""


# Second pass: Generate additional charts to cover unexplored aspects
EXPLORATORY_CHART_PROMPT = """You are a data visualization expert. Review what has already been visualized and generate ADDITIONAL charts to cover aspects not yet addressed.

The code will run in a sandboxed environment with these PYTHON VARIABLES already defined:
- DATA_DIR: String variable containing the absolute path to CSV files
- OUTPUT_DIR: String variable containing the absolute path to save figures

IMPORTANT: DATA_DIR and OUTPUT_DIR are ALREADY DEFINED Python string variables.
Do NOT use os.environ.get() or define them yourself. Just use them directly.

=== ALREADY CREATED CHARTS ===
{existing_charts}

=== COMPUTED STATISTICS ===
(Use these as ground truth for any new visualizations)

REQUIREMENTS:
1. Generate 3-4 ADDITIONAL charts that cover different aspects of the data
2. DO NOT recreate charts similar to what already exists
3. Consider: distributions, correlations, time trends, outliers, subgroup analyses
4. Each chart should provide distinct value

Return JSON with this structure:
{
    "charts": [
        {
            "chart_id": "fig_descriptive_name",
            "title": "Human-readable title",
            "chart_type": "bar|line|box|scatter|histogram|heatmap|violin",
            "insight": "What this chart reveals",
            "ground_truth_values": {"key": "value pairs from computed stats"},
            "code": "complete python code string"
        }
    ]
}

RULES FOR EACH CHART:
1. matplotlib is already imported with Agg backend
2. Use pd.read_csv(f"{DATA_DIR}/filename.csv")
3. Save as plt.savefig(f"{OUTPUT_DIR}/{chart_id}.png", dpi=300, bbox_inches='tight')
4. Always plt.close() at the end
5. Handle data cleaning (dropna, type conversion)
6. Use try/except for robustness
7. ALLOWED LIBRARIES: pandas, numpy, matplotlib, seaborn, scipy, statsmodels
8. DO NOT use any other libraries (no plotly, no sklearn, etc)"""

def analyze_raw_context(
    client: GeminiLLMClient,
    raw_context: str,
    thinking_level: str = "high"
) -> str:
    """
    Analyze raw context files (code, specs) to generate a methodology summary.
    
    Args:
        client: Gemini client
        raw_context: The verbatim content from raw_context files
        thinking_level: Chain-of-thought depth
        
    Returns:
        Detailed summary of the raw context for use by all agents
    """
    if not raw_context or len(raw_context.strip()) < 100:
        return "No significant raw context available for analysis."
    
    response = client.call_text(
        system_prompt=RAW_CONTEXT_ANALYSIS_PROMPT,
        user_text=f"Analyze this code/documentation and create a comprehensive methodology summary:\n\n{raw_context}",
        thinking_level=thinking_level
    )
    
    return response


def compute_statistics(
    client: GeminiLLMClient,
    data_passport: str,
    data_dir: str,
    output_dir: str,
    user_guidance: str = "",
    methodology_summary: str = "",
    thinking_level: str = "high"
) -> str:
    """
    Step 1 of compute-then-visualize: Generate and execute code to compute key statistics.
    
    Uses a "best-of-N" approach:
    1. Generate code attempt 1 → Execute
    2. Generate code attempt 2 → Execute  
    3. Select the best output from both attempts
    
    Args:
        client: Gemini client
        data_passport: Smart data passport with samples
        data_dir: Path to raw data CSVs
        output_dir: Directory to save computed_stats.md
        user_guidance: User instructions for stats computation
        methodology_summary: Summary of raw context (code/docs)
        thinking_level: Chain-of-thought depth
        
    Returns:
        Contents of computed_stats.md (the ground truth statistics)
    """
    from pathlib import Path
    
    tools_dir = str(Path(output_dir).parent / "tools")
    
    # Format system prompt with guidance and methodology context
    methodology_text = methodology_summary if methodology_summary else "No methodology summary available."
    
    system_prompt = STATS_COMPUTATION_PROMPT.replace(
        "{user_guidance}", 
        user_guidance if user_guidance else "No specific user guidance provided."
    ).replace(
        "{methodology_summary}",
        methodology_text
    )
    
    user_text = f"Analyze this data structure and generate code to compute comprehensive statistics:\n\n{data_passport}"
    
    attempts = []
    
    # === ATTEMPT 1 ===
    log_event("INFO", "Analyst", "Generating stats code (attempt 1)")
    code1 = client.call_text(
        system_prompt=system_prompt,
        user_text=user_text,
        thinking_level=thinking_level
    )
    code1 = _clean_code_block(code1)
    save_code_artifact(code1, "stats_computation", 1, tools_dir)
    
    # Execute attempt 1 to a temp file
    output1_file = Path(output_dir) / "computed_stats_attempt1.md"
    result1 = execute_python_code(
        code1.replace("computed_stats.md", "computed_stats_attempt1.md"),
        output_dir, data_dir
    )
    
    if result1.success and output1_file.exists():
        output1 = output1_file.read_text(encoding="utf-8")
        attempts.append({"attempt": 1, "success": True, "output": output1, "error": None})
        log_event("INFO", "Analyst", f"Attempt 1 succeeded: {len(output1)} chars")
    else:
        attempts.append({"attempt": 1, "success": False, "output": "", "error": result1.error})
        log_event("WARNING", "Analyst", f"Attempt 1 failed: {result1.error[:100] if result1.error else 'Unknown'}")
    
    # === ATTEMPT 2 ===
    log_event("INFO", "Analyst", "Generating stats code (attempt 2)")
    code2 = client.call_text(
        system_prompt=system_prompt,
        user_text=user_text + "\n\nProvide a different approach if possible.",
        thinking_level=thinking_level
    )
    code2 = _clean_code_block(code2)
    save_code_artifact(code2, "stats_computation", 2, tools_dir)
    
    # Execute attempt 2 to a temp file
    output2_file = Path(output_dir) / "computed_stats_attempt2.md"
    result2 = execute_python_code(
        code2.replace("computed_stats.md", "computed_stats_attempt2.md"),
        output_dir, data_dir
    )
    
    if result2.success and output2_file.exists():
        output2 = output2_file.read_text(encoding="utf-8")
        attempts.append({"attempt": 2, "success": True, "output": output2, "error": None})
        log_event("INFO", "Analyst", f"Attempt 2 succeeded: {len(output2)} chars")
    else:
        attempts.append({"attempt": 2, "success": False, "output": "", "error": result2.error})
        log_event("WARNING", "Analyst", f"Attempt 2 failed: {result2.error[:100] if result2.error else 'Unknown'}")
    
    # === SELECTION ===
    successful_attempts = [a for a in attempts if a["success"]]
    
    if len(successful_attempts) == 0:
        # Both failed
        log_event("ERROR", "Analyst", "Both stats computation attempts failed")
        return "Statistics computation failed in both attempts. Refer to data passport for available statistics."
    
    if len(successful_attempts) == 1:
        # Only one succeeded, use it
        best = successful_attempts[0]
        log_event("INFO", "Analyst", f"Only attempt {best['attempt']} succeeded, using it")
        final_output = best["output"]
    else:
        # Both succeeded, use API to select best
        log_event("INFO", "Analyst", "Both attempts succeeded, selecting best output")
        
        selection_prompt = """You are a statistics reviewer. Compare these two statistical outputs and select the BETTER one.

CRITERIA FOR SELECTION:
1. Completeness: More comprehensive statistics and tests
2. Accuracy: Correct use of statistical methods (e.g., proper percentile keys like '50%' not 'median')
3. Formatting: Clear, well-organized markdown tables
4. Error-free: No runtime errors or missing values

Return ONLY the number 1 or 2 to indicate which attempt is better."""

        selection_text = f"""=== ATTEMPT 1 ===
{attempts[0]['output'][:8000]}

=== ATTEMPT 2 ===
{attempts[1]['output'][:8000]}

Which is better? Return only 1 or 2."""

        selection = client.call_text(
            system_prompt=selection_prompt,
            user_text=selection_text,
            thinking_level="low"
        ).strip()
        
        # Parse selection
        if "2" in selection and "1" not in selection.replace("21", ""):
            best_idx = 1
        else:
            best_idx = 0  # Default to attempt 1 if unclear
        
        best = attempts[best_idx]
        log_event("INFO", "Analyst", f"Selected attempt {best['attempt']} as best")
        final_output = best["output"]
    
    # Write final output to computed_stats.md
    final_file = Path(output_dir) / "computed_stats.md"
    final_file.write_text(final_output, encoding="utf-8")
    log_event("INFO", "Analyst", f"Computed statistics: {len(final_output)} chars")
    
    return final_output


def _clean_code_block(code: str) -> str:
    """Remove markdown code fences from generated code."""
    if not code:
        return ""
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()


def plan_and_generate_charts(
    client: GeminiLLMClient,
    data_passport: str,
    computed_stats: str,
    user_guidance: str = "",
    methodology_summary: str = "",
    thinking_level: str = "high"
) -> List[Dict[str, Any]]:
    """
    Step 2 of compute-then-visualize: Plan AND generate all chart code in one API call.
    
    This receives computed statistics as ground truth, ensuring charts match the data.
    
    Args:
        client: Gemini client
        data_passport: Smart data passport with samples
        computed_stats: Pre-computed statistics (ground truth)
        user_guidance: User instructions from inputs/instructions.md
        methodology_summary: Summary of raw context
        thinking_level: Chain-of-thought depth
        
    Returns:
        List of chart specifications with embedded code
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
                        "title": {"type": "string"},
                        "chart_type": {"type": "string"},
                        "insight": {"type": "string"},
                        "ground_truth_values": {"type": "string"},
                        "code": {"type": "string"}
                    },
                    "required": ["chart_id", "title", "code"]
                }
            }
        },
        "required": ["charts"]
    }
    
    # Format system prompt with guidance
    methodology_text = methodology_summary if methodology_summary else "No methodology summary available."
    
    # Use replace() instead of format() to avoid issues with JSON braces
    system_prompt = BATCHED_CHART_PROMPT.replace(
        "{user_guidance}", 
        user_guidance if user_guidance else "No specific user guidance provided."
    ).replace(
        "{methodology_summary}",
        methodology_text
    )
    
    prompt = f"""Generate charts for this data.

=== COMPUTED STATISTICS (GROUND TRUTH) ===
{computed_stats}

=== DATA STRUCTURE ===
{data_passport}

Create multiple charts that visualize the key findings. Each chart's data must match the computed statistics above."""
    
    response = client.call_text(
        system_prompt=system_prompt,
        user_text=prompt,
        thinking_level=thinking_level,
        response_schema=response_schema
    )
    
    data = json.loads(response)
    charts = data.get("charts", [])
    log_event("INFO", "Analyst", f"Generated {len(charts)} chart specifications with code")
    return charts


def generate_exploratory_charts(
    client: GeminiLLMClient,
    data_passport: str,
    computed_stats: str,
    existing_charts: List[Dict[str, Any]],
    thinking_level: str = "high"
) -> List[Dict[str, Any]]:
    """
    Second pass: Generate additional charts to cover aspects not yet addressed.
    
    Args:
        client: Gemini client
        data_passport: Smart data passport with samples
        computed_stats: Pre-computed statistics (ground truth)
        existing_charts: Charts already generated in first pass
        thinking_level: Chain-of-thought depth
        
    Returns:
        List of additional chart specifications with embedded code
    """
    # Build summary of existing charts
    existing_summary = "\n".join([
        f"- {c.get('chart_id', 'unknown')}: {c.get('title', 'Untitled')} ({c.get('chart_type', 'unknown')})"
        for c in existing_charts
    ])
    
    response_schema = {
        "type": "object",
        "properties": {
            "charts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "chart_id": {"type": "string"},
                        "title": {"type": "string"},
                        "chart_type": {"type": "string"},
                        "insight": {"type": "string"},
                        "ground_truth_values": {"type": "string"},
                        "code": {"type": "string"}
                    },
                    "required": ["chart_id", "title", "code"]
                }
            }
        },
        "required": ["charts"]
    }
    
    # Format system prompt with existing charts context
    system_prompt = EXPLORATORY_CHART_PROMPT.replace(
        "{existing_charts}", 
        existing_summary if existing_summary else "No charts created yet."
    )
    
    prompt = f"""Generate additional charts for this data.

=== COMPUTED STATISTICS (GROUND TRUTH) ===
{computed_stats}

=== DATA STRUCTURE ===
{data_passport}

Create 3-4 additional charts that explore aspects not covered by the existing charts."""
    
    response = client.call_text(
        system_prompt=system_prompt,
        user_text=prompt,
        thinking_level=thinking_level,
        response_schema=response_schema
    )
    
    data = json.loads(response)
    charts = data.get("charts", [])
    log_event("INFO", "Analyst", f"Generated {len(charts)} exploratory chart specifications")
    return charts


def generate_chart_variations_batch(
    client: GeminiLLMClient,
    data_passport: str,
    original_charts: List[Dict[str, Any]],
    thinking_level: str = "high"
) -> List[Dict[str, Any]]:
    """
    Generate alternative code for a batch of charts (Batch "Attempt 2").
    
    Args:
        client: Gemini client
        data_passport: Data schema
        original_charts: The list of charts from Attempt 1
        thinking_level: Depth of thought
        
    Returns:
        List of "Attempt 2" chart specifications
    """
    if not original_charts:
        return []
        
    # Create summary of what we already have - including the V1 code so LLM can improve it
    chart_summaries = []
    for c in original_charts:
        chart_summaries.append({
            "chart_id": c["chart_id"],
            "title": c["title"],
            "current_type": c["chart_type"],
            "insight": c["insight"],
            "v1_code": c.get("code", "")  # Include V1 code so LLM knows what to improve
        })
        
    response_schema = {
        "type": "object",
        "properties": {
            "charts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "chart_id": {"type": "string"},
                        "code": {"type": "string"}
                    },
                    "required": ["chart_id", "code"]
                }
            }
        },
        "required": ["charts"]
    }
    
    prompt = f"""You are a Python visualization expert and quality reviewer.
    
We have generated an initial set of charts ("Attempt 1"). Now we need "Attempt 2" - an IMPROVED/REFINED version for each chart.

The code will run in a sandboxed environment with these PYTHON VARIABLES already defined:
- DATA_DIR: String variable containing the absolute path to CSV files
- OUTPUT_DIR: String variable containing the absolute path to save figures

=== DATA ===
{data_passport}

=== CHARTS TO IMPROVE (including V1 code) ===
{json.dumps(chart_summaries, indent=2)}

TASK:
For EACH chart_id above, review the "v1_code" provided and generate IMPROVED Python code that creates a BETTER version.

FOCUS ON IMPROVEMENTS TO THE V1 CODE:
- Better axis labels, titles, and legends (clear and descriptive)
- Improved color schemes (publication-quality, colorblind-friendly)
- Better data presentation (appropriate scales, no clipping)
- Add annotations for key values where helpful
- Ensure proper figure sizing and margins
- Add error bars or confidence intervals if applicable
- Fix any potential edge cases (empty data, missing values)
- Fix any bugs or issues in the V1 code

RULES FOR EACH CHART CODE:
1. matplotlib is already imported with Agg backend
2. Use pd.read_csv(f"{{DATA_DIR}}/filename.csv") to load data. DO NOT assume 'df' exists.
3. Save as plt.savefig(f"{{OUTPUT_DIR}}/{{chart_id}}_v2.png", dpi=300, bbox_inches='tight')
   - Note: append '_v2' to the filename!
4. Always plt.close() at the end
5. DO NOT use plt.show()
6. Handle data cleaning (dropna, type conversion)
7. ALLOWED LIBRARIES: pandas, numpy, matplotlib, seaborn, scipy, statsmodels
8. if using patheffects, use `from matplotlib.patheffects import withStroke` (NOT matplotlib.patheffects directly)
9. Keep code simple - avoid complex visual effects that might fail

Return JSON mapping chart_id to new code."""

    response = client.call_text(
        system_prompt="Generate improved Python visualization code with better quality and presentation.",
        user_text=prompt,
        thinking_level=thinking_level,
        response_schema=response_schema
    )
    
    data = json.loads(response)
    variations = data.get("charts", [])
    
    # Merge variation code into full chart objects (inheriting metadata from original)
    final_variations = []
    for var in variations:
        original = next((c for c in original_charts if c["chart_id"] == var["chart_id"]), None)
        if original:
            new_chart = original.copy()
            new_chart["code"] = var["code"]
            # Mark title as Variation
            new_chart["title"] = original["title"] + " (Variation)"
            final_variations.append(new_chart)
            
    log_event("INFO", "Analyst", f"Generated {len(final_variations)} chart variations")
    return final_variations


def select_best_charts_batch(
    client: GeminiLLMClient,
    pairs: List[Dict[str, Any]],
    thinking_level: str = "low"
) -> List[str]:
    """
    Use Vision to select the best chart from pairs (Original vs Variation).
    
    Args:
        client: Gemini client
        pairs: List of dicts {chart_id, v1_path, v2_path}
        thinking_level: Low for speed
        
    Returns:
        List of selected image paths (one per chart_id)
    """
    if not pairs:
        return []
        
    # Prepare prompt
    prompt = "You are a Design Editor. For each pair of charts, select the one that is CLEARER, MORE AESTHETIC, PUBLISHABLE, and BETTER COMMUNICATES the data.\n\n"
    image_paths = []
    
    for i, pair in enumerate(pairs):
        cid = pair["chart_id"]
        v1 = pair["v1_path"]
        v2 = pair["v2_path"]
        
        prompt += f"Pair {i+1}: {cid}\n"
        prompt += f"- A: First image ({Path(v1).name})\n"
        prompt += f"- B: Second image ({Path(v2).name})\n\n"
        
        image_paths.append(v1)
        image_paths.append(v2)
        
    prompt += """Evaluate each pair.
Return JSON:
{
    "selections": [
        {"chart_id": "fig_1", "winner": "A"},
        {"chart_id": "fig_2", "winner": "B"}
    ]
}"""

    response_schema = {
        "type": "object",
        "properties": {
            "selections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "chart_id": {"type": "string"},
                        "winner": {"type": "string", "enum": ["A", "B"]}
                    },
                    "required": ["chart_id", "winner"]
                }
            }
        },
        "required": ["selections"]
    }

    try:
        response = client.call_images(
            system_prompt="Select the best data visualization from each pair.",
            user_text=prompt,
            image_paths=image_paths,
            thinking_level=thinking_level,
            response_schema=response_schema
        )
        data = json.loads(response)
        selections = data.get("selections", [])
        
        results = []
        for sel in selections:
            cid = sel["chart_id"]
            winner = sel["winner"]
            pair = next((p for p in pairs if p["chart_id"] == cid), None)
            if pair:
                selected_path = pair["v1_path"] if winner == "A" else pair["v2_path"]
                results.append(selected_path)
                log_event("INFO", "Analyst", f"Selected {winner} for {cid}")
                
        return results
        
    except Exception as e:
        log_event("ERROR", "Analyst", f"Vision selection failed: {e}. Defaulting to V1s.")
        return [p["v1_path"] for p in pairs]


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
    return _clean_code_block(code)


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


def interpret_charts_batch(
    client: GeminiLLMClient,
    figure_paths: List[str],
    thinking_level: str = "low"
) -> str:
    """
    Use Vision to interpret ALL charts in a single API call.
    
    Args:
        client: Gemini client
        figure_paths: List of paths to chart images
        thinking_level: Chain-of-thought depth (low for efficiency)
        
    Returns:
        Combined interpretation text for all figures
    """
    if not figure_paths:
        return "No figures to interpret."
    
    # Build descriptive prompt with figure names
    figure_names = [Path(p).stem for p in figure_paths]
    user_text = f"Analyze these {len(figure_paths)} figures: {', '.join(figure_names)}"
    
    response = client.call_images(
        system_prompt=BATCH_VISION_PROMPT,
        user_text=user_text,
        image_paths=figure_paths,
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
    
    # Analyze raw context first (code files, methodology specs)
    raw_context = state.get("raw_context", "")
    knowledge_base_dir = str(Path(run_dir) / "00_Knowledge_Base")
    Path(knowledge_base_dir).mkdir(parents=True, exist_ok=True)
    
    if raw_context:
        update_display("Analyst", "Analyzing raw context (code/specs)...")
        log_event("INFO", "Analyst", "Analyzing raw context files")
        raw_context_summary = analyze_raw_context(client, raw_context, thinking_level)
        state["raw_context_summary"] = raw_context_summary
        log_event("INFO", "Analyst", f"Raw context summary: {len(raw_context_summary)} chars")
        
        # Save to file for debugging and transparency
        summary_file = Path(knowledge_base_dir) / "methodology_summary.md"
        summary_file.write_text(raw_context_summary, encoding="utf-8")
        log_event("INFO", "Analyst", f"Saved methodology summary to {summary_file}")
    else:
        state["raw_context_summary"] = "No raw context files provided."
    
    # Get data passport from evidence items or load fresh
    # Also determine the absolute path to raw_data
    raw_data_dir = str(Path.cwd() / "inputs" / "raw_data")
    
    if state.get("evidence_items"):
        # Build passport from evidence
        passport_parts = []
        for ev_id, item in state["evidence_items"].items():
            passport_parts.append(f"=== {ev_id}: {item.source_file} ===\n{item.data_snippet}\n")
            # Get absolute path from first evidence item
            if not raw_data_dir or raw_data_dir == str(Path.cwd() / "inputs" / "raw_data"):
                raw_data_dir = str(Path(item.source_file).parent.resolve())
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
        state["computed_statistics"] = ""
        return state
    
    # === NEW COMPUTE-THEN-VISUALIZE FLOW ===
    
    # Step 1: Compute statistics first (1 API call)
    update_display("Analyst", "Computing key statistics...")
    knowledge_base_dir = str(Path(run_dir) / "00_Knowledge_Base")
    Path(knowledge_base_dir).mkdir(parents=True, exist_ok=True)
    
    # Load granular guidance
    guidance_stats = state.get("guidance_stats", "")
    if guidance_stats:
        log_event("INFO", "Analyst", "Applying user guidance to stats computation")
        
    # Get methodology summary for context
    methodology_summary = state.get("raw_context_summary", "")

    computed_stats = compute_statistics(
        client, data_passport, raw_data_dir, knowledge_base_dir, guidance_stats, methodology_summary, thinking_level
    )
    state["computed_statistics"] = computed_stats
    log_event("INFO", "Analyst", f"Computed stats: {len(computed_stats)} chars")
    
    # Step 2: Generate all charts with computed stats (1 API call)
    update_display("Analyst", "Planning and generating all charts...")
    
    guidance_charts = state.get("guidance_charts", "")
    if guidance_charts:
        log_event("INFO", "Analyst", "Applying user guidance to chart planning")
        
    charts = plan_and_generate_charts(client, data_passport, computed_stats, guidance_charts, methodology_summary, thinking_level)
    log_event("INFO", "Analyst", f"Generated {len(charts)} primary chart specifications")
    
    # Step 2b: Generate exploratory charts (1 additional API call)
    update_display("Analyst", "Generating exploratory charts...")
    exploratory_charts = generate_exploratory_charts(
        client, data_passport, computed_stats, charts, thinking_level
    )
    log_event("INFO", "Analyst", f"Generated {len(exploratory_charts)} exploratory chart specifications")
    
    # Step 2c: Generate Variations (Best-of-N) - 1 API Call
    update_display("Analyst", "Generating chart variations (Best-of-N)...")
    variations = generate_chart_variations_batch(client, data_passport, charts, thinking_level)
    
    # Combine original charts and variations for execution
    # Map chart_id to its variations
    chart_pairs = [] # {chart_id, v1, v2}
    
    all_execution_charts = []
    
    for original in charts:
        cid = original["chart_id"]
        # Find variation
        variation = next((v for v in variations if v["chart_id"] == cid), None)
        
        # Add original (V1)
        original_v1 = original.copy()
        original_v1["chart_id"] = f"{cid}_v1"
        # Patch the code to save with _v1 suffix instead of original name
        if "code" in original_v1 and original_v1["code"]:
            original_v1["code"] = original_v1["code"].replace(f"{cid}.png", f"{cid}_v1.png")
        all_execution_charts.append(original_v1)
        
        pair = {"chart_id": cid, "v1_id": f"{cid}_v1", "v1_chart": original_v1, "v2_id": None, "v2_chart": None}
        
        if variation:
            # Add variation (V2)
            variation_v2 = variation.copy()
            variation_v2["chart_id"] = f"{cid}_v2"
            all_execution_charts.append(variation_v2)
            pair["v2_id"] = f"{cid}_v2"
            pair["v2_chart"] = variation_v2
            
        chart_pairs.append(pair)
            
    # Add exploratory charts (processed normally, single attempt)
    for exp in exploratory_charts:
        all_execution_charts.append(exp)
        
    log_event("INFO", "Analyst", f"Total charts to execute (including variations): {len(all_execution_charts)}")
    
    figure_manifest = {}
    successful_charts = []  # Collect for batch vision later
    
    # Track execution results for Best-of-N selection
    execution_results = {} # chart_id -> fig_path
    
    # Step 3: Execute each chart
    for chart in all_execution_charts:
        chart_id = chart.get("chart_id", "unknown")
        code = chart.get("code", "")
        
        if not code:
            continue
            
        update_display("Analyst", f"Executing {chart_id}...")
        
        # Clean up code
        code = _clean_code_block(code)
        
        attempt = 1
        success = False
        
        while attempt <= max_retries and not success:
            code_path = save_code_artifact(code, chart_id, attempt, tools_dir)
            
            # Execute
            result = execute_python_code(code, figures_dir, raw_data_dir)
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
                    execution_results[chart_id] = fig_path
                    
                    # If this is an exploratory chart (not in pairs), add directly to successful_charts
                    if any(e["chart_id"] == chart_id for e in exploratory_charts):
                        log_event("INFO", "Analyst", f"Successfully generated exploratory {chart_id}")
                        successful_charts.append({
                            "chart_id": chart_id,
                            "fig_path": fig_path,
                            "code_path": code_path,
                            "title": chart.get("title", chart_id),
                            "ground_truth": chart.get("ground_truth_values", {})
                        })
            else:
                # Self-correction
                if attempt < max_retries:
                    update_display("Analyst", f"Fixing {chart_id} (attempt {attempt+1})...")
                    code = fix_code_error(client, code, result.error, thinking_level)
                attempt += 1
                
    # Step 4: Select Best-of-N (1 API Call)
    update_display("Analyst", "Selecting best visualizations...")
    
    pairs_for_selection = []
    selection_log = []  # Track selections for audit
    
    for pair in chart_pairs:
        cid = pair["chart_id"]
        v1_path = execution_results.get(pair["v1_id"])
        v2_path = execution_results.get(pair["v2_id"])
        
        if v1_path and v2_path:
            # Both succeeded - add to selection batch
            pairs_for_selection.append({
                "chart_id": cid,
                "v1_path": v1_path,
                "v2_path": v2_path
            })
        elif v1_path:
            # Only V1 succeeded - automatic winner
            log_event("INFO", "Analyst", f"Only V1 succeeded for {cid}, using it")
            final_path = str(Path(figures_dir) / f"{cid}.png")
            try:
                import shutil
                time.sleep(1)  # Wait for file handles to release on Windows
                shutil.copy2(v1_path, final_path)
                
                selection_log.append({
                    "chart_id": cid,
                    "winner": "V1 (only option)",
                    "source": Path(v1_path).name,
                    "final": Path(final_path).name
                })
                
                successful_charts.append({
                    "chart_id": cid,
                    "fig_path": final_path,
                    "code_path": str(Path(tools_dir) / f"{pair['v1_id']}_code.py"),
                    "title": pair["v1_chart"]["title"],
                    "ground_truth": pair["v1_chart"].get("ground_truth_values", {})
                })
            except Exception as e:
                log_event("ERROR", "Analyst", f"Failed to copy file: {e}")
                selection_log.append({
                    "chart_id": cid, "winner": "FAILED", "source": "error", "final": str(e)
                })
                
        elif v2_path:
            # Only V2 succeeded - automatic winner
            log_event("INFO", "Analyst", f"Only V2 succeeded for {cid}, using it")
            final_path = str(Path(figures_dir) / f"{cid}.png")
            try:
                import shutil
                time.sleep(1)  # Wait for file handles to release on Windows
                shutil.copy2(v2_path, final_path)
                
                selection_log.append({
                    "chart_id": cid,
                    "winner": "V2 (only option)",
                    "source": Path(v2_path).name,
                    "final": Path(final_path).name
                })
                
                successful_charts.append({
                    "chart_id": cid,
                    "fig_path": final_path,
                    "code_path": str(Path(tools_dir) / f"{pair['v2_id']}_code.py"),
                    "title": pair["v2_chart"]["title"],
                    "ground_truth": pair["v2_chart"].get("ground_truth_values", {})
                })
            except Exception as e:
                log_event("ERROR", "Analyst", f"Failed to copy file: {e}")
                selection_log.append({
                    "chart_id": cid, "winner": "FAILED", "source": "error", "final": str(e)
                })
        else:
            # Both failed
            selection_log.append({
                "chart_id": cid, "winner": "NONE", "source": "both failed", "final": "N/A"
            })
    
    # Run Batch Selection for pairs where both V1 and V2 succeeded
    if pairs_for_selection:
        selected_paths = select_best_charts_batch(client, pairs_for_selection, thinking_level="low")
        
        # Important: Wait for Windows to release file handles after Vision API read them
        time.sleep(10)
        
        for i, selected_path in enumerate(selected_paths):
            pair_info = pairs_for_selection[i]
            cid = pair_info["chart_id"]
            
            final_path = str(Path(figures_dir) / f"{cid}.png")
            try:
                import shutil
                shutil.copy2(selected_path, final_path)
                
                is_v1 = (selected_path == pair_info["v1_path"])
                winner_label = "V1" if is_v1 else "V2"
                
                selection_log.append({
                    "chart_id": cid,
                    "winner": winner_label,
                    "source": Path(selected_path).name,
                    "final": Path(final_path).name
                })
                
                original_pair = next((p for p in chart_pairs if p["chart_id"] == cid), None)
                if original_pair:
                    winner_chart = original_pair["v1_chart"] if is_v1 else original_pair["v2_chart"]
                    
                    successful_charts.append({
                        "chart_id": cid,
                        "fig_path": final_path,
                        "code_path": str(Path(tools_dir) / f"{original_pair['v1_id' if is_v1 else 'v2_id']}_code.py"),
                        "title": winner_chart["title"],
                        "ground_truth": winner_chart.get("ground_truth_values", {})
                    })
            except Exception as e:
                log_event("ERROR", "Analyst", f"Failed to process selection for {cid}: {e}")
                selection_log.append({
                    "chart_id": cid, "winner": "FAILED", "source": "error", "final": str(e)
                })
    
    # Write selection log
    log_path = Path(figures_dir) / "selection_log.md"
    log_content = "# Best-of-N Chart Selection Log\n\n"
    log_content += "| Chart ID | Winner | Source File | Final File |\n"
    log_content += "|----------|--------|-------------|------------|\n"
    for entry in selection_log:
        log_content += f"| {entry['chart_id']} | {entry['winner']} | {entry['source']} | {entry['final']} |\n"
    log_path.write_text(log_content, encoding="utf-8")
    log_event("INFO", "Analyst", f"Saved selection log to {log_path}")
    
    # Step 4: Batch vision analysis (1 API call for all charts)
    interpretations = []
    if successful_charts:
        figure_paths = [c["fig_path"] for c in successful_charts]
        update_display("Analyst", f"Analyzing {len(figure_paths)} figures with Vision...")
        log_event("INFO", "Analyst", f"Batch vision analysis for {len(figure_paths)} figures")
        
        batch_interpretation = interpret_charts_batch(client, figure_paths, thinking_level="low")
        
        # Create figure manifest with chart metadata (using insight from chart spec instead of vision)
        for chart_info in successful_charts:
            chart_id = chart_info["chart_id"]
            
            figure_manifest[chart_id] = FigureItem(
                figure_id=chart_id,
                file_path=chart_info["fig_path"],
                code_path=chart_info["code_path"],
                interpretation=f"Figure: {chart_info['title']}",
                suggested_caption=chart_info["title"],
                data_source=str(chart_info["ground_truth"]) if chart_info["ground_truth"] else "raw_data"
            )
            
            interpretations.append(f"### {chart_info['title']}")
        
        # Add batch vision interpretation at the end
        interpretations.append("\n## Vision Analysis\n" + batch_interpretation)
        
        # Save vision analysis to file
        vision_file = Path(knowledge_base_dir) / "vision_analysis.md"
        vision_file.write_text(batch_interpretation, encoding="utf-8")
        log_event("INFO", "Analyst", f"Saved vision analysis to {vision_file}")
    
    # Compile analysis context with computed stats included
    analysis_context = "# Data Analysis Report\n\n"
    analysis_context += "## Computed Statistics\n\n"
    analysis_context += computed_stats + "\n\n"
    analysis_context += f"## Visualizations\n\nGenerated {len(figure_manifest)} figures.\n\n"
    analysis_context += "\n\n".join(interpretations)
    
    # Save full analysis context to file
    context_file = Path(knowledge_base_dir) / "analysis_context.md"
    context_file.write_text(analysis_context, encoding="utf-8")
    log_event("INFO", "Analyst", f"Saved analysis context to {context_file}")
    
    state["analysis_context"] = analysis_context
    state["figure_manifest"] = figure_manifest
    
    log_event("INFO", "Analyst", f"Completed analysis with {len(figure_manifest)} figures")
    
    return state

