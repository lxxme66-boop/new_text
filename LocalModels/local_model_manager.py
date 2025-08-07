"""
本地模型管理器
用于管理和选择不同的本地模型后端
"""
import logging
from typing import Dict, Any, Optional, Union

from .ollama_client import OllamaClient, create_ollama_client

# 尝试导入vLLM
try:
    from .vllm_client import VLLMClient, create_vllm_client
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

logger = logging.getLogger(__name__)


class LocalModelManager:
    """本地模型管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化模型管理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.models = {}
        self.default_backend = config.get('default_backend', 'ollama')
        
    def get_client(self, backend: Optional[str] = None):
        """
        获取指定后端的客户端
        
        Args:
            backend: 后端名称 ('ollama', 'vllm')
            
        Returns:
            对应的客户端实例
        """
        if backend is None:
            backend = self.default_backend
            
        if backend in self.models:
            return self.models[backend]
            
        if backend == 'ollama':
            client = create_ollama_client(self.config.get('ollama', {}))
            self.models[backend] = client
            return client
            
        elif backend == 'vllm' and VLLM_AVAILABLE:
            client = create_vllm_client(self.config.get('vllm', {}))
            self.models[backend] = client
            return client
            
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    
    async def generate(self, prompt: str, backend: Optional[str] = None, **kwargs):
        """
        使用指定后端生成文本
        
        Args:
            prompt: 输入提示词
            backend: 后端名称
            **kwargs: 额外参数
            
        Returns:
            生成的文本
        """
        client = self.get_client(backend)
        
        if hasattr(client, 'agenerate'):
            return await client.agenerate(prompt, **kwargs)
        else:
            # 同步方法的异步包装
            import asyncio
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, client.generate, prompt, **kwargs)
    
    def list_available_backends(self):
        """列出可用的后端"""
        backends = ['ollama']
        if VLLM_AVAILABLE:
            backends.append('vllm')
        return backends


def create_local_model_manager(config: Optional[Dict[str, Any]] = None) -> LocalModelManager:
    """
    创建本地模型管理器
    
    Args:
        config: 配置字典
        
    Returns:
        LocalModelManager实例
    """
    if config is None:
        config = {
            'default_backend': 'ollama',
            'ollama': {
                'base_url': 'http://localhost:11434',
                'model': 'qwen2.5:32b'
            }
        }
        
        if VLLM_AVAILABLE:
            config['vllm'] = {
                'model_path': '/mnt/workspace/models/Qwen/QwQ-32B/',
                'gpu_memory_utilization': 0.95,
                'tensor_parallel_size': 1
            }
    
    return LocalModelManager(config)