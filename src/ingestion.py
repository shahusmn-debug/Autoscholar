# src/ingestion.py
"""
Dual-Path Ingestion Pipeline

This module handles loading and processing input data:
- Path A (PDF/Vision): Uses Gemini's native PDF understanding via Files API
- Path B (Text): Processes .txt and .md notes directly
- Raw Data: Loads CSV files and creates data passports
- User Draft: Loads existing papers for refinement mode
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

from .client import GeminiLLMClient
from .state import ReferenceItem, EvidenceItem
from .progress import log_event, update_display


# System prompts for ingestion
PDF_EXTRACTION_PROMPT = """You are an expert academic research assistant.
Analyze this PDF document and extract:
1. Title and authors
2. Year of publication
3. A comprehensive summary (2-3 paragraphs)
4. Key findings (as a list)
5. Methodology overview
6. Relevance to research synthesis

Return your analysis as JSON with this schema:
{
    "title": "string",
    "authors": "string",
    "year": "string",
    "summary": "string",
    "key_findings": ["string"],
    "methodology": "string"
}"""

TEXT_EXTRACTION_PROMPT = """You are an expert academic research assistant.
Analyze this text document and extract key information.
If this appears to be research notes, summarize the main points.
If this is a paper excerpt, identify claims and evidence.

Return your analysis as JSON with this schema:
{
    "summary": "string",
    "key_findings": ["string"],
    "relevance_notes": "string"
}"""


def ingest_pdf(
    pdf_path: str,
    client: GeminiLLMClient,
    ref_id: str,
    transcripts_dir: str,
    thinking_level: str = "high"
) -> ReferenceItem:
    """
    Ingest a PDF using Gemini's native PDF understanding (Path A).
    
    This uses the Files API to upload the PDF, avoiding OCR errors
    by leveraging Gemini's multimodal PDF processing.
    
    Args:
        pdf_path: Path to the PDF file
        client: Initialized GeminiLLMClient
        ref_id: Unique reference identifier
        transcripts_dir: Directory to save raw LLM transcripts
        thinking_level: Chain-of-thought depth
        
    Returns:
        ReferenceItem with extracted information
    """
    update_display("Ingestion", f"Processing PDF: {Path(pdf_path).name}")
    log_event("INFO", "Ingestion", f"Starting PDF ingestion: {pdf_path}")
    
    # Define response schema for structured output
    response_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "authors": {"type": "string"},
            "year": {"type": "string"},
            "summary": {"type": "string"},
            "key_findings": {"type": "array", "items": {"type": "string"}},
            "methodology": {"type": "string"}
        },
        "required": ["summary", "key_findings"]
    }
    
    try:
        # Call Gemini with the PDF
        response = client.call_pdf_via_files_api(
            system_prompt=PDF_EXTRACTION_PROMPT,
            user_text="Extract information from this academic paper.",
            pdf_path=pdf_path,
            thinking_level=thinking_level,
            response_schema=response_schema
        )
        
        # Parse JSON response
        data = json.loads(response)
        
        # Save transcript
        transcript_path = Path(transcripts_dir) / f"ingestion_{ref_id}.json"
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump({
                "input": {"pdf_path": pdf_path},
                "output": data
            }, f, indent=2)
        
        log_event("INFO", "Ingestion", f"Successfully extracted from PDF: {data.get('title', 'Unknown')}")
        
        return ReferenceItem(
            ref_id=ref_id,
            source_file=pdf_path,
            title=data.get("title"),
            authors=data.get("authors"),
            year=data.get("year"),
            summary=data.get("summary", ""),
            key_findings=data.get("key_findings", []),
            methodology=data.get("methodology"),
            relevance_notes=""
        )
        
    except Exception as e:
        log_event("ERROR", "Ingestion", f"Failed to process PDF {pdf_path}: {str(e)}")
        # Return partial reference on error
        return ReferenceItem(
            ref_id=ref_id,
            source_file=pdf_path,
            summary=f"[Extraction failed: {str(e)}]"
        )


def ingest_text(
    text_path: str,
    client: GeminiLLMClient,
    ref_id: str,
    transcripts_dir: str,
    thinking_level: str = "high"
) -> ReferenceItem:
    """
    Ingest a text or markdown file (Path B).
    
    For lighter-weight notes and text documents.
    
    Args:
        text_path: Path to the .txt or .md file
        client: Initialized GeminiLLMClient
        ref_id: Unique reference identifier
        transcripts_dir: Directory to save transcripts
        thinking_level: Chain-of-thought depth
        
    Returns:
        ReferenceItem with extracted information
    """
    update_display("Ingestion", f"Processing text: {Path(text_path).name}")
    log_event("INFO", "Ingestion", f"Starting text ingestion: {text_path}")
    
    # Read the file
    with open(text_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    response_schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "key_findings": {"type": "array", "items": {"type": "string"}},
            "relevance_notes": {"type": "string"}
        },
        "required": ["summary"]
    }
    
    try:
        response = client.call_text(
            system_prompt=TEXT_EXTRACTION_PROMPT,
            user_text=f"Analyze this document:\n\n{content}",
            thinking_level=thinking_level,
            response_schema=response_schema
        )
        
        data = json.loads(response)
        
        # Save transcript
        transcript_path = Path(transcripts_dir) / f"ingestion_{ref_id}.json"
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump({
                "input": {"text_path": text_path, "content_preview": content[:500]},
                "output": data
            }, f, indent=2)
        
        log_event("INFO", "Ingestion", f"Successfully processed text file: {text_path}")
        
        return ReferenceItem(
            ref_id=ref_id,
            source_file=text_path,
            summary=data.get("summary", ""),
            key_findings=data.get("key_findings", []),
            relevance_notes=data.get("relevance_notes", "")
        )
        
    except Exception as e:
        log_event("ERROR", "Ingestion", f"Failed to process text {text_path}: {str(e)}")
        return ReferenceItem(
            ref_id=ref_id,
            source_file=text_path,
            summary=f"[Extraction failed: {str(e)}]"
        )


def load_raw_data(data_dir: str) -> Tuple[Dict[str, EvidenceItem], str]:
    """
    Load CSV files and create data passports.
    
    A data passport includes:
    - Column names and types
    - First 50 rows as sample
    - Basic statistics
    
    Args:
        data_dir: Path to the raw_data directory
        
    Returns:
        Tuple of (evidence items dict, data passport text)
    """
    update_display("Ingestion", "Loading raw data files")
    log_event("INFO", "Ingestion", f"Scanning raw data directory: {data_dir}")
    
    data_path = Path(data_dir)
    evidence_items = {}
    passport_parts = []
    
    # Process CSV files
    csv_files = list(data_path.glob("*.csv"))
    
    for i, csv_file in enumerate(csv_files):
        try:
            df = pd.read_csv(csv_file)
            ev_id = f"ev_{i+1:03d}"
            
            # Create data passport
            passport = f"""
=== Data File: {csv_file.name} ===
Rows: {len(df)}, Columns: {len(df.columns)}
Columns: {', '.join(df.columns.tolist())}

Sample (first 50 rows):
{df.head(50).to_string()}

Statistics:
{df.describe().to_string()}
"""
            passport_parts.append(passport)
            
            # Create evidence item
            evidence_items[ev_id] = EvidenceItem(
                evidence_id=ev_id,
                source_file=str(csv_file),
                description=f"Data from {csv_file.name}",
                data_snippet=df.head(5).to_string(),
                statistical_summary=df.describe().to_string()
            )
            
            log_event("INFO", "Ingestion", f"Loaded CSV: {csv_file.name} ({len(df)} rows)")
            
        except Exception as e:
            log_event("ERROR", "Ingestion", f"Failed to load {csv_file.name}: {str(e)}")
    
    data_passport = "\n".join(passport_parts) if passport_parts else "No CSV data files found."
    
    return evidence_items, data_passport


def load_user_draft(draft_dir: str) -> Optional[str]:
    """
    Load an existing draft for refinement mode.
    
    Looks for draft.md in the user_draft directory.
    
    Args:
        draft_dir: Path to inputs/user_draft
        
    Returns:
        Draft content as string, or None if not found
    """
    draft_path = Path(draft_dir) / "draft.md"
    
    if not draft_path.exists():
        log_event("WARNING", "Ingestion", "No draft.md found in user_draft directory")
        return None
    
    with open(draft_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    word_count = len(content.split())
    log_event("INFO", "Ingestion", f"Loaded user draft: {word_count} words")
    
    return content


def ingest_all_references(
    references_dir: str,
    client: GeminiLLMClient,
    transcripts_dir: str,
    thinking_level: str = "high"
) -> Dict[str, ReferenceItem]:
    """
    Ingest all reference files from the references directory.
    
    Routes files to appropriate handler:
    - .pdf → ingest_pdf (Path A)
    - .txt, .md → ingest_text (Path B)
    
    Args:
        references_dir: Path to inputs/references
        client: Initialized GeminiLLMClient
        transcripts_dir: Directory for transcripts
        thinking_level: Chain-of-thought depth
        
    Returns:
        Dict of ref_id → ReferenceItem
    """
    ref_path = Path(references_dir)
    knowledge_base = {}
    ref_counter = 1
    
    # Get all reference files
    pdf_files = list(ref_path.glob("*.pdf"))
    txt_files = list(ref_path.glob("*.txt")) + list(ref_path.glob("*.md"))
    
    total_files = len(pdf_files) + len(txt_files)
    log_event("INFO", "Ingestion", f"Found {total_files} reference files to process")
    
    # Process PDFs (Path A)
    for pdf_file in pdf_files:
        ref_id = f"Ref_{ref_counter:03d}"
        ref_item = ingest_pdf(
            str(pdf_file), client, ref_id, transcripts_dir, thinking_level
        )
        knowledge_base[ref_id] = ref_item
        ref_counter += 1
    
    # Process text files (Path B)
    for txt_file in txt_files:
        ref_id = f"Ref_{ref_counter:03d}"
        ref_item = ingest_text(
            str(txt_file), client, ref_id, transcripts_dir, thinking_level
        )
        knowledge_base[ref_id] = ref_item
        ref_counter += 1
    
    log_event("INFO", "Ingestion", f"Completed ingestion of {len(knowledge_base)} references")
    
    return knowledge_base
