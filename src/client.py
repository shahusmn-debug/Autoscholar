# src/client.py
"""
Gemini LLM Client - Wrapper for Google Gemini API

This module provides a unified interface to Google's Gemini API with support for:
- Text generation with thinking levels
- Vision/image analysis
- PDF processing via Files API
- Structured JSON outputs
"""

import os
import mimetypes
from typing import Any, Optional, Dict, List

from google import genai
from google.genai import types


class GeminiLLMClient:
    """
    A wrapper class for interacting with Google's Gemini AI models.
    
    This client handles both text and vision models, supporting:
    - Different API versions for text (v1beta) and vision (v1alpha)
    - Thinking levels for chain-of-thought reasoning
    - Structured JSON response schemas
    - File uploads via the Files API
    
    Attributes:
        client_text: Gemini client configured for text generation
        client_vision: Gemini client configured for vision tasks
        model_text: The text model name (e.g., "gemini-2.5-pro-preview-06-05")
        model_vision: The vision model name
    """
    
    def __init__(
        self,
        model_text: str,
        model_vision: str,
        api_version_text: str = "v1beta",
        api_version_vision: str = "v1alpha"
    ):
        """
        Initialize the Gemini client with specified models and API versions.
        
        Args:
            model_text: Model name for text generation
            model_vision: Model name for vision/image tasks
            api_version_text: API version for text (default: v1beta)
            api_version_vision: API version for vision (default: v1alpha)
            
        Raises:
            RuntimeError: If no API key is found in environment variables
        """
        # Get API key from environment
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        if not api_key and not os.getenv("GOOGLE_GENAI_USE_VERTEXAI"):
            raise RuntimeError(
                "Missing GEMINI_API_KEY or GOOGLE_API_KEY environment variable. "
                "Please set one of these, or configure Vertex AI environment variables."
            )
        
        # Create separate clients for text and vision with different API versions
        self.client_text = genai.Client(
            api_key=api_key,
            http_options={"api_version": api_version_text}
        )
        self.client_vision = genai.Client(
            api_key=api_key,
            http_options={"api_version": api_version_vision}
        )
        
        self.model_text = model_text
        self.model_vision = model_vision
    
    def call_text(
        self,
        system_prompt: str,
        user_text: str,
        thinking_level: str = "high",
        response_schema: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate text using the Gemini text model.
        
        Args:
            system_prompt: The system instruction that sets the AI's behavior
            user_text: The user's input/query
            thinking_level: Chain-of-thought depth ("low", "medium", "high")
            response_schema: Optional JSON schema for structured output
            
        Returns:
            The generated text response
        """
        # Build configuration
        cfg = types.GenerateContentConfig(
            system_instruction=system_prompt,
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
        )
        
        # Add structured output if schema provided
        if response_schema is not None:
            cfg.response_mime_type = "application/json"
            cfg.response_schema = response_schema
        
        # Make the API call
        resp = self.client_text.models.generate_content(
            model=self.model_text,
            contents=user_text,
            config=cfg,
        )
        
        return resp.text
    
    def call_images(
        self,
        system_prompt: str,
        user_text: str,
        image_paths: List[str],
        thinking_level: str = "high",
        media_resolution_level: Optional[str] = None,
        response_schema: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Analyze images using Gemini's vision capabilities.
        
        Args:
            system_prompt: The system instruction
            user_text: The user's query about the images
            image_paths: List of file paths to images
            thinking_level: Chain-of-thought depth
            media_resolution_level: Optional resolution setting for images
            response_schema: Optional JSON schema for structured output
            
        Returns:
            The generated analysis text
        """
        # Build parts list starting with the text prompt
        parts = [types.Part(text=user_text)]
        
        # Add each image as a part
        for path in image_paths:
            # Detect MIME type
            mime, _ = mimetypes.guess_type(path)
            if not mime:
                mime = "image/png"  # Default fallback
            
            # Read image data
            with open(path, "rb") as f:
                data = f.read()
            
            # Create blob
            blob = types.Blob(mime_type=mime, data=data)
            part_kwargs = {"inline_data": blob}
            
            # Add media resolution if specified (v1alpha only)
            if media_resolution_level:
                part_kwargs["media_resolution"] = {"level": media_resolution_level}
            
            parts.append(types.Part(**part_kwargs))
        
        # Build configuration
        cfg = types.GenerateContentConfig(
            system_instruction=system_prompt,
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
        )
        
        if response_schema is not None:
            cfg.response_mime_type = "application/json"
            cfg.response_schema = response_schema
        
        # Make the API call with vision client
        resp = self.client_vision.models.generate_content(
            model=self.model_vision,
            contents=[types.Content(parts=parts)],
            config=cfg,
        )
        
        return resp.text
    
    def upload_file(self, path: str):
        """
        Upload a file to the Gemini Files API.
        
        The Files API is useful for large files like PDFs that exceed
        the inline data limit.
        
        Args:
            path: Path to the file to upload
            
        Returns:
            The uploaded file handle from the API
        """
        return self.client_text.files.upload(file=path)
    
    def call_pdf_via_files_api(
        self,
        system_prompt: str,
        user_text: str,
        pdf_path: str,
        thinking_level: str = "high",
        response_schema: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Process a PDF document using the Files API.
        
        This method uploads the PDF first, then includes it in the
        generation request. This is more reliable for large PDFs
        and avoids OCR errors by using Gemini's native PDF understanding.
        
        Args:
            system_prompt: The system instruction
            user_text: The user's query about the PDF
            pdf_path: Path to the PDF file
            thinking_level: Chain-of-thought depth
            response_schema: Optional JSON schema for structured output
            
        Returns:
            The generated response text
        """
        # Upload the PDF file first
        uploaded = self.upload_file(pdf_path)
        
        # Build configuration
        cfg = types.GenerateContentConfig(
            system_instruction=system_prompt,
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
        )
        
        if response_schema is not None:
            cfg.response_mime_type = "application/json"
            cfg.response_schema = response_schema
        
        # Make the API call with the uploaded file
        resp = self.client_text.models.generate_content(
            model=self.model_text,
            contents=[uploaded, user_text],
            config=cfg,
        )
        
        return resp.text
