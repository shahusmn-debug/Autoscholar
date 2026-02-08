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


# Batched extraction prompts - process multiple documents in one call
BATCHED_PDF_EXTRACTION_PROMPT = """You are an expert academic research assistant.
Analyze ALL of the attached PDF documents and extract key information from each.

For EACH PDF document, extract:
1. Title and authors
2. Year of publication  
3. A comprehensive summary (2-3 paragraphs)
4. Key findings (as a list)
5. Methodology overview

Return your analysis as a JSON array with one object per PDF, in the same order they were provided:
{
    "documents": [
        {
            "title": "string",
            "authors": "string", 
            "year": "string",
            "summary": "string",
            "key_findings": ["string"],
            "methodology": "string"
        }
    ]
}"""


BATCHED_TEXT_EXTRACTION_PROMPT = """You are an expert academic research assistant.
Analyze ALL of the text documents below and extract key information from each.

For EACH document section (marked with === headers), extract:
1. Summary of main points
2. Key findings
3. Relevance notes

Return your analysis as a JSON array with one object per document, in the same order they were provided:
{
    "documents": [
        {
            "filename": "string",
            "summary": "string",
            "key_findings": ["string"],
            "relevance_notes": "string"
        }
    ]
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


def ingest_pdfs_batched(
    pdf_paths: List[str],
    client: GeminiLLMClient,
    transcripts_dir: str,
    thinking_level: str = "high"
) -> Dict[str, ReferenceItem]:
    """
    Ingest multiple PDFs in a SINGLE API call to reduce request count.
    
    Args:
        pdf_paths: List of paths to PDF files
        client: Initialized GeminiLLMClient
        transcripts_dir: Directory to save raw LLM transcripts
        thinking_level: Chain-of-thought depth
        
    Returns:
        Dict of ref_id → ReferenceItem for all PDFs
    """
    if not pdf_paths:
        return {}
    
    update_display("Ingestion", f"Processing {len(pdf_paths)} PDFs (batched)...")
    log_event("INFO", "Ingestion", f"Batched PDF ingestion: {len(pdf_paths)} files")
    
    # Build response schema for array of documents
    response_schema = {
        "type": "object",
        "properties": {
            "documents": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "authors": {"type": "string"},
                        "year": {"type": "string"},
                        "summary": {"type": "string"},
                        "key_findings": {"type": "array", "items": {"type": "string"}},
                        "methodology": {"type": "string"}
                    },
                    "required": ["summary"]
                }
            }
        },
        "required": ["documents"]
    }
    
    try:
        # Single API call for all PDFs
        response = client.call_multi_pdf_via_files_api(
            system_prompt=BATCHED_PDF_EXTRACTION_PROMPT,
            user_text=f"Extract information from these {len(pdf_paths)} academic papers.",
            pdf_paths=pdf_paths,
            thinking_level=thinking_level,
            response_schema=response_schema
        )
        
        data = json.loads(response)
        documents = data.get("documents", [])
        
        # Build reference items
        knowledge_base = {}
        for i, (pdf_path, doc_data) in enumerate(zip(pdf_paths, documents), start=1):
            ref_id = f"Ref_{i:03d}"
            
            ref_item = ReferenceItem(
                ref_id=ref_id,
                source_file=pdf_path,
                title=doc_data.get("title"),
                authors=doc_data.get("authors"),
                year=doc_data.get("year"),
                summary=doc_data.get("summary", ""),
                key_findings=doc_data.get("key_findings", []),
                methodology=doc_data.get("methodology")
            )
            knowledge_base[ref_id] = ref_item
            log_event("INFO", "Ingestion", f"Extracted: {doc_data.get('title', Path(pdf_path).name)}")
        
        # Save transcript
        transcript_path = Path(transcripts_dir) / "ingestion_pdfs_batched.json"
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump({"input": pdf_paths, "output": data}, f, indent=2)
        
        return knowledge_base
        
    except Exception as e:
        log_event("ERROR", "Ingestion", f"Batched PDF ingestion failed: {str(e)}")
        # Fallback: process individually
        log_event("INFO", "Ingestion", "Falling back to individual PDF processing")
        knowledge_base = {}
        for i, pdf_path in enumerate(pdf_paths, start=1):
            ref_id = f"Ref_{i:03d}"
            ref_item = ingest_pdf(pdf_path, client, ref_id, transcripts_dir, thinking_level)
            knowledge_base[ref_id] = ref_item
        return knowledge_base


def ingest_texts_batched(
    text_paths: List[str],
    client: GeminiLLMClient,
    transcripts_dir: str,
    ref_start_id: int = 1,
    thinking_level: str = "high"
) -> Dict[str, ReferenceItem]:
    """
    Ingest multiple text files in a SINGLE API call to reduce request count.
    
    Args:
        text_paths: List of paths to text/markdown files
        client: Initialized GeminiLLMClient
        transcripts_dir: Directory to save raw LLM transcripts
        ref_start_id: Starting number for ref_id (continues from PDFs)
        thinking_level: Chain-of-thought depth
        
    Returns:
        Dict of ref_id → ReferenceItem for all text files
    """
    if not text_paths:
        return {}
    
    update_display("Ingestion", f"Processing {len(text_paths)} text files (batched)...")
    log_event("INFO", "Ingestion", f"Batched text ingestion: {len(text_paths)} files")
    
    # Concatenate all text files with headers
    combined_content = []
    for text_path in text_paths:
        with open(text_path, 'r', encoding='utf-8') as f:
            content = f.read()
        filename = Path(text_path).name
        combined_content.append(f"=== Document: {filename} ===\n{content}")
    
    full_text = "\n\n".join(combined_content)
    
    # Build response schema
    response_schema = {
        "type": "object",
        "properties": {
            "documents": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"},
                        "summary": {"type": "string"},
                        "key_findings": {"type": "array", "items": {"type": "string"}},
                        "relevance_notes": {"type": "string"}
                    },
                    "required": ["summary"]
                }
            }
        },
        "required": ["documents"]
    }
    
    try:
        # Single API call for all text files
        response = client.call_text(
            system_prompt=BATCHED_TEXT_EXTRACTION_PROMPT,
            user_text=f"Analyze these {len(text_paths)} documents:\n\n{full_text}",
            thinking_level=thinking_level,
            response_schema=response_schema
        )
        
        data = json.loads(response)
        documents = data.get("documents", [])
        
        # Build reference items
        knowledge_base = {}
        for i, (text_path, doc_data) in enumerate(zip(text_paths, documents), start=ref_start_id):
            ref_id = f"Ref_{i:03d}"
            
            ref_item = ReferenceItem(
                ref_id=ref_id,
                source_file=text_path,
                summary=doc_data.get("summary", ""),
                key_findings=doc_data.get("key_findings", []),
                relevance_notes=doc_data.get("relevance_notes", "")
            )
            knowledge_base[ref_id] = ref_item
            log_event("INFO", "Ingestion", f"Processed text: {Path(text_path).name}")
        
        # Save transcript
        transcript_path = Path(transcripts_dir) / "ingestion_texts_batched.json"
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump({"input": text_paths, "output": data}, f, indent=2)
        
        return knowledge_base
        
    except Exception as e:
        log_event("ERROR", "Ingestion", f"Batched text ingestion failed: {str(e)}")
        # Fallback: process individually
        log_event("INFO", "Ingestion", "Falling back to individual text processing")
        knowledge_base = {}
        for i, text_path in enumerate(text_paths, start=ref_start_id):
            ref_id = f"Ref_{i:03d}"
            ref_item = ingest_text(text_path, client, ref_id, transcripts_dir, thinking_level)
            knowledge_base[ref_id] = ref_item
        return knowledge_base


def load_raw_data(data_dir: str) -> Tuple[Dict[str, EvidenceItem], str]:
    """
    Load CSV files and create smart data passports.
    
    A smart data passport includes:
    - File metadata (rows, columns)
    - Column names and data types
    - Sample rows (10 rows, with long text truncated)
    - Full statistics from describe()
    
    This provides structure visibility without context explosion from long text fields.
    
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
    
    # Configuration for smart passport
    SAMPLE_ROWS = 10
    MAX_TEXT_LENGTH = 100  # Truncate long text columns
    
    # Process CSV files
    csv_files = list(data_path.glob("*.csv"))
    
    for i, csv_file in enumerate(csv_files):
        try:
            df = pd.read_csv(csv_file)
            
            # --- SANITIZATION START ---
            # Strip whitespace from column names to avoid "New " vs "New" issues
            original_cols = list(df.columns)
            df.columns = df.columns.str.strip()
            
            # Check if cleaning happened
            if list(df.columns) != original_cols:
                log_event("INFO", "Ingestion", f"Sanitized column names in {csv_file.name} (stripped whitespace)")
                
                # Save cleaned version to ensure Analyst code reads clean data
                # We prefix with 'clean_' to avoid overwriting original data
                clean_filename = f"clean_{csv_file.name}"
                clean_path = csv_file.parent / clean_filename
                df.to_csv(clean_path, index=False)
                
                # Update reference to use the CLEAN file
                csv_file = clean_path
            # --- SANITIZATION END ---
            ev_id = f"ev_{i+1:03d}"
            
            # Create sample with truncated text columns
            sample_df = df.head(SAMPLE_ROWS).copy()
            for col in sample_df.columns:
                if sample_df[col].dtype == 'object':
                    # Truncate string columns
                    sample_df[col] = sample_df[col].apply(
                        lambda x: (str(x)[:MAX_TEXT_LENGTH] + '...') if isinstance(x, str) and len(str(x)) > MAX_TEXT_LENGTH else x
                    )
            
            # Build data types info
            dtypes_info = "\n".join([f"  - {col}: {dtype}" for col, dtype in df.dtypes.items()])
            
            # Create smart data passport
            passport = f"""
=== Data File: {csv_file.name} ===
Shape: {len(df)} rows × {len(df.columns)} columns

Column Types:
{dtypes_info}

Sample Data ({SAMPLE_ROWS} rows, text truncated to {MAX_TEXT_LENGTH} chars):
{sample_df.to_string()}

Statistics:
{df.describe(include='all').to_string()}
"""
            passport_parts.append(passport)
            
            # Create evidence item
            evidence_items[ev_id] = EvidenceItem(
                evidence_id=ev_id,
                source_file=str(csv_file.resolve()),  # Use absolute path
                description=f"Data from {csv_file.name}",
                data_snippet=sample_df.to_string(),
                statistical_summary=df.describe(include='all').to_string()
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


def load_raw_context(raw_context_dir: str) -> str:
    """
    Load raw context files VERBATIM without AI summarization.
    
    This is for code, specifications, or other content that should
    be passed directly to the writer without being processed by AI.
    Files are concatenated with headers indicating their source.
    
    Args:
        raw_context_dir: Path to inputs/raw_context
        
    Returns:
        Concatenated content of all files with source headers
    """
    context_path = Path(raw_context_dir)
    
    if not context_path.exists():
        log_event("INFO", "Ingestion", "No raw_context directory found (optional)")
        return ""
    
    parts = []
    supported_extensions = {'.txt', '.md', '.py', '.r', '.sql', '.json', '.yaml', '.yml', '.csv'}
    
    for file_path in sorted(context_path.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                parts.append(f"=== RAW CONTEXT: {file_path.name} ===\n{content}\n")
                log_event("INFO", "Ingestion", f"Loaded raw context: {file_path.name} ({len(content)} chars)")
            except Exception as e:
                log_event("WARNING", "Ingestion", f"Failed to load raw context {file_path.name}: {e}")
    
    if parts:
        log_event("INFO", "Ingestion", f"Loaded {len(parts)} raw context files")
    
    return "\n".join(parts)


def ingest_all_references(
    references_dir: str,
    client: GeminiLLMClient,
    transcripts_dir: str,
    thinking_level: str = "high"
) -> Dict[str, ReferenceItem]:
    """
    Ingest all reference files from the references directory.
    
    Uses BATCHED ingestion to minimize API calls:
    - All PDFs processed in 1 API call
    - All text files processed in 1 API call
    - Fallback to individual processing if batch fails
    
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
    
    # Get all reference files
    pdf_files = [str(f) for f in ref_path.glob("*.pdf")]
    txt_files = [str(f) for f in ref_path.glob("*.txt")] + [str(f) for f in ref_path.glob("*.md")]
    
    total_files = len(pdf_files) + len(txt_files)
    log_event("INFO", "Ingestion", f"Found {total_files} reference files to process")
    
    # Process PDFs (batched - 1 API call for all)
    if pdf_files:
        pdf_refs = ingest_pdfs_batched(
            pdf_files, client, transcripts_dir, thinking_level
        )
        knowledge_base.update(pdf_refs)
    
    # Process text files (batched - 1 API call for all)
    if txt_files:
        # Start ref IDs after PDFs
        ref_start_id = len(pdf_files) + 1
        txt_refs = ingest_texts_batched(
            txt_files, client, transcripts_dir, ref_start_id, thinking_level
        )
        knowledge_base.update(txt_refs)
    
    log_event("INFO", "Ingestion", f"Completed ingestion of {len(knowledge_base)} references")
    
    return knowledge_base


def load_instructions(inputs_dir: str) -> tuple[str, str]:
    """
    Load user guidance from inputs/user_instructions directory.
    
    Structure:
    inputs/user_instructions/
        stats_compute/  -> guidance_stats
        figure_gen/     -> guidance_charts
        
    Each folder can contain multiple .txt or .md files.
    """
    from pathlib import Path
    
    base_path = Path(inputs_dir) / "user_instructions"
    stats_path = base_path / "stats_compute"
    charts_path = base_path / "figure_gen"
    
    guidance_stats = ""
    guidance_charts = ""
    
    # helper to read all files in a dir
    def read_dir(p: Path) -> str:
        content = []
        if p.exists() and p.is_dir():
            for f in sorted(p.glob("*")):
                if f.is_file() and f.suffix in ['.txt', '.md']:
                    try:
                        text = f.read_text(encoding="utf-8")
                        content.append(f"--- FILE: {f.name} ---\n{text}")
                    except Exception as e:
                        log_event("WARNING", "Ingestion", f"Failed to read instruction file {f.name}: {e}")
        return "\n\n".join(content)
        
    # Standard loading from folders
    if base_path.exists():
        guidance_stats = read_dir(stats_path)
        guidance_charts = read_dir(charts_path)
        
    # Legacy fallback (optional, but good for stability during transition)
    legacy_file = Path(inputs_dir) / "instructions.md"
    if legacy_file.exists():
        legacy_content = legacy_file.read_text(encoding="utf-8")
        if not guidance_charts:
            guidance_charts = legacy_content
            log_event("INFO", "Ingestion", "Loaded legacy instructions.md as chart guidance")
        else:
            guidance_charts += "\n\n--- LEGACY instructions.md ---\n" + legacy_content
            
    # Log results
    if guidance_stats:
        log_event("INFO", "Ingestion", f"Loaded stats guidance ({len(guidance_stats)} chars)")
    else:
        log_event("INFO", "Ingestion", "No stats computation guidance found")
        
    if guidance_charts:
        log_event("INFO", "Ingestion", f"Loaded chart guidance ({len(guidance_charts)} chars)")
    else:
        log_event("INFO", "Ingestion", "No chart generation guidance found")
        
    return guidance_stats, guidance_charts

