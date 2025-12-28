#!/usr/bin/env python3
# main.py
"""
AutoScholar-MVP - Autonomous Research Agent

CLI entry point for the research paper generation pipeline.

Usage:
    python main.py --topic "Impact of X on Y"           # Create mode
    python main.py --mode refine                        # Refine existing draft
    python main.py --help                               # Show help

Modes:
    create (default): Generate paper from scratch using references and data
    refine: Improve an existing draft in inputs/user_draft/draft.md
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AutoScholar-MVP: Autonomous Research Paper Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --topic "Effects of climate change on crop yields"
  python main.py --mode refine --topic "My paper about AI ethics"
  
Directory Structure:
  inputs/
    ├── references/     # Place PDF papers and text notes here
    ├── raw_data/       # Place CSV data files here
    └── user_draft/     # Place draft.md here (for refine mode)
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["create", "refine"],
        default="create",
        help="Operation mode: 'create' for new paper, 'refine' for editing existing (default: create)"
    )
    
    parser.add_argument(
        "--topic",
        type=str,
        default="",
        help="Research topic for the paper (required for create mode)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    
    parser.add_argument(
        "--inputs",
        type=str,
        default="inputs",
        help="Path to inputs directory (default: inputs)"
    )
    
    parser.add_argument(
        "--runs",
        type=str,
        default="runs",
        help="Path to runs output directory (default: runs)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup without making API calls"
    )
    
    return parser.parse_args()


def validate_setup(args) -> bool:
    """
    Validate that the environment is properly configured.
    
    Returns:
        True if setup is valid, False otherwise
    """
    import os
    
    errors = []
    warnings = []
    
    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key and not os.getenv("GOOGLE_GENAI_USE_VERTEXAI"):
        errors.append("Missing GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
    
    # Check config file
    config_path = Path(args.config)
    if not config_path.exists():
        errors.append(f"Config file not found: {args.config}")
    
    # Check inputs directory
    inputs_path = Path(args.inputs)
    if not inputs_path.exists():
        errors.append(f"Inputs directory not found: {args.inputs}")
    else:
        # Check subdirectories
        if not (inputs_path / "references").exists():
            warnings.append("inputs/references directory not found")
        if not (inputs_path / "raw_data").exists():
            warnings.append("inputs/raw_data directory not found")
        
        # For refine mode, check for draft
        if args.mode == "refine":
            draft_path = inputs_path / "user_draft" / "draft.md"
            if not draft_path.exists():
                warnings.append("Refine mode but no draft.md found in inputs/user_draft/")
    
    # Check for topic in create mode
    if args.mode == "create" and not args.topic:
        warnings.append("No --topic specified for create mode (will use generic prompts)")
    
    # Print results
    from src.progress import print_error, print_warning, print_success, print_info
    
    if errors:
        for error in errors:
            print_error(error)
        return False
    
    if warnings:
        for warning in warnings:
            print_warning(warning)
    
    print_success("Setup validation passed")
    return True


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load environment variables
    from src.utils import load_dotenv_if_exists
    load_dotenv_if_exists()
    
    # Import progress utilities
    from src.progress import print_header, print_success, print_error, print_info
    
    # Print header
    print_header("AutoScholar-MVP")
    print_info(f"Mode: {args.mode}")
    if args.topic:
        print_info(f"Topic: {args.topic}")
    
    # Validate setup
    if not validate_setup(args):
        print_error("Setup validation failed. Please fix the errors above.")
        sys.exit(1)
    
    # Dry run mode
    if args.dry_run:
        print_success("Dry run complete. Setup is valid.")
        sys.exit(0)
    
    # Load config
    from src.utils import load_config
    config = load_config(args.config)
    
    # Run pipeline
    try:
        from src.graph import run_pipeline
        
        print_info("Starting pipeline...")
        
        final_path = run_pipeline(
            mode=args.mode,
            topic=args.topic,
            config=config,
            inputs_dir=args.inputs,
            runs_dir=args.runs
        )
        
        print_success(f"Pipeline complete!")
        print_info(f"Final paper: {final_path}")
        
    except KeyboardInterrupt:
        print_error("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
