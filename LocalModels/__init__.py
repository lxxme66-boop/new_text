"""
Local Models Module

This module provides support for local large language models.
It includes functionality for:
- Ollama model integration
- vLLM model serving
- Transformers local model loading
- Custom model API interfaces
- Model performance optimization
"""

from .ollama_client import OllamaClient, create_ollama_client
from .vllm_client import VLLMClient, create_vllm_client
from .transformers_client import TransformersClient, create_transformers_client
from .local_model_manager import LocalModelManager, get_available_models
from .model_utils import ModelUtils, estimate_model_size

__all__ = [
    'OllamaClient',
    'create_ollama_client',
    'VLLMClient', 
    'create_vllm_client',
    'TransformersClient',
    'create_transformers_client',
    'LocalModelManager',
    'get_available_models',
    'ModelUtils',
    'estimate_model_size'
]

__version__ = "1.0.0"