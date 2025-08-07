"""
MultiModal Module

This module provides multimodal processing capabilities for text and image data.
It includes functionality for:
- PDF document processing and image extraction
- Multimodal QA generation
- Image-text alignment and analysis
- Professional domain-specific multimodal processing
"""

from .pdf_processor import PDFProcessor, process_pdf_folder
from .multimodal_datageneration import MultiModalDataGenerator, parse_pdf
from .image_utils import ImageProcessor, encode_image
from .multimodal_qa_generator import MultiModalQAGenerator

__all__ = [
    'PDFProcessor',
    'process_pdf_folder',
    'MultiModalDataGenerator',
    'parse_pdf',
    'ImageProcessor',
    'encode_image',
    'MultiModalQAGenerator'
]

__version__ = "1.0.0"