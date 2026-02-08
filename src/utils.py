# src/utils.py
"""
Utility Functions

Helper functions used across the AutoScholar pipeline.
"""

import os
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_run_directory(base_dir: str = "runs") -> str:
    """
    Create a timestamped directory for this run's outputs.
    
    Creates structure:
    runs/Job_{timestamp}/
    ├── 00_Knowledge_Base/
    ├── figures/
    ├── tables/
    ├── transcripts/
    └── tools/
    
    Args:
        base_dir: Base directory for runs
        
    Returns:
        Path to the created run directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"Job_{timestamp}"
    
    # Create main directory and subdirectories
    subdirs = [
        "00_Knowledge_Base",
        "figures",
        "tables",
        "transcripts",
        "tools"
    ]
    
    for subdir in subdirs:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Create empty log files
    (run_dir / "run.log").touch()
    (run_dir / "events.jsonl").touch()
    
    return str(run_dir)


def get_section_dir(run_dir: str, section_name: str) -> str:
    """
    Get the directory path for a specific section.
    
    Args:
        run_dir: The run directory path
        section_name: Name of the section (e.g., "Introduction")
        
    Returns:
        Path to the section directory
    """
    # Section map matches the order in state.py section_queue
    section_map = {
        "Methodology": "00_Methodology",
        "Results": "01_Results",
        "Discussion": "02_Discussion",
        "Introduction": "03_Introduction",
        "Literature Review": "04_Literature_Review",
        "Conclusion": "05_Conclusion"
    }
    
    subdir = section_map.get(section_name, section_name.replace(" ", "_"))
    return str(Path(run_dir) / subdir)


def save_iteration_draft(
    run_dir: str,
    section: str,
    iteration: int,
    draft: str,
    critique: Optional[str] = None
) -> str:
    """
    Save a draft and optional critique for an iteration.
    
    Creates:
    {section_dir}/iter_{n}/
    ├── draft.md
    └── critique.md (if provided)
    
    Args:
        run_dir: The run directory path
        section: Section name
        iteration: Iteration number
        draft: The draft content
        critique: Optional critique/feedback
        
    Returns:
        Path to the saved draft
    """
    section_dir = Path(get_section_dir(run_dir, section))
    iter_dir = section_dir / f"iter_{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    
    # Save draft
    draft_path = iter_dir / "draft.md"
    with open(draft_path, 'w', encoding='utf-8') as f:
        f.write(draft)
    
    # Save critique if provided
    if critique:
        critique_path = iter_dir / "critique.md"
        with open(critique_path, 'w', encoding='utf-8') as f:
            f.write(critique)
    
    return str(draft_path)


def save_final_paper(run_dir: str, content: str) -> str:
    """
    Save the final assembled paper.
    
    Args:
        run_dir: The run directory path
        content: The final paper content
        
    Returns:
        Path to FINAL_PAPER.md
    """
    final_path = Path(run_dir) / "FINAL_PAPER.md"
    with open(final_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return str(final_path)


def save_knowledge_base(run_dir: str, knowledge_base: dict, analysis: str) -> None:
    """
    Save the knowledge base to the run directory.
    
    Args:
        run_dir: The run directory path
        knowledge_base: Dict of ReferenceItems
        analysis: The analysis report text
    """
    kb_dir = Path(run_dir) / "00_Knowledge_Base"
    kb_dir.mkdir(parents=True, exist_ok=True)
    
    # Save evidence notes as JSON
    evidence_path = kb_dir / "evidence_notes.json"
    
    # Convert dataclass items to dicts
    kb_dict = {}
    for ref_id, item in knowledge_base.items():
        if hasattr(item, '__dict__'):
            kb_dict[ref_id] = item.__dict__
        else:
            kb_dict[ref_id] = dict(item)
    
    with open(evidence_path, 'w', encoding='utf-8') as f:
        import json
        json.dump(kb_dict, f, indent=2, default=str)
    
    # Save analysis report
    analysis_path = kb_dir / "analysis_report.md"
    with open(analysis_path, 'w', encoding='utf-8') as f:
        f.write(analysis)


def load_dotenv_if_exists() -> None:
    """Load .env file if it exists."""
    try:
        from dotenv import load_dotenv
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass  # python-dotenv not installed
