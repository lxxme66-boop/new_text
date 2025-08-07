#!/usr/bin/env python3
"""
vLLM客户端 - 支持在多GPU上运行大模型
针对2卡A100优化
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
import subprocess
import time
import os

logger = logging.getLogger(__name__)


class VLLMClient:
    """vLLM本地模型客户端"""
    
    def __init__(self, config: Dict):
        """
        初始化vLLM客户端
        
        Args:
            config: vLLM配置
        """
        self.config = config
        self.base_url = f"http://localhost:{config.get('port', 8000)}"
        self.model_path = config.get('model_path')
        self.tensor_parallel_size = config.get('tensor_parallel_size', 2)
        self.gpu_memory_utilization = config.get('gpu_memory_utilization', 0.9)
        self.max_model_len = config.get('max_model_len', 8192)
        self.dtype = config.get('dtype', 'float16')
        self.session = None
        self.server_process = None
        
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.start_server()
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
        if self.server_process:
            self.stop_server()
    
    async def start_server(self):
        """启动vLLM服务器"""
        if self.check_server_running():
            logger.info("vLLM服务器已在运行")
            return
        
        logger.info("启动vLLM服务器...")
        
        # 构建启动命令
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--max-model-len", str(self.max_model_len),
            "--dtype", self.dtype,
            "--port", str(self.config.get('port', 8000)),
            "--host", "0.0.0.0"
        ]
        
        # 如果配置了trust_remote_code
        if self.config.get('trust_remote_code', False):
            cmd.append("--trust-remote-code")
        
        # 设置环境变量以使用指定的GPU
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0,1'  # 使用GPU 0和1
        
        # 启动服务器进程
        self.server_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 等待服务器启动
        max_retries = 30
        for i in range(max_retries):
            if self.check_server_running():
                logger.info("vLLM服务器启动成功")
                return
            await asyncio.sleep(2)
        
        raise RuntimeError("vLLM服务器启动失败")
    
    def stop_server(self):
        """停止vLLM服务器"""
        if self.server_process:
            logger.info("停止vLLM服务器...")
            self.server_process.terminate()
            self.server_process.wait()
            self.server_process = None
    
    def check_server_running(self) -> bool:
        """检查服务器是否在运行"""
        try:
            import requests
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        url = f"{self.base_url}/v1/completions"
        
        payload = {
            "model": self.model_path,
            "prompt": prompt,
            "max_tokens": kwargs.get('max_tokens', 2048),
            "temperature": kwargs.get('temperature', 0.8),
            "top_p": kwargs.get('top_p', 0.9),
            "frequency_penalty": kwargs.get('frequency_penalty', 0.0),
            "presence_penalty": kwargs.get('presence_penalty', 0.0),
            "stream": False
        }
        
        async with self.session.post(url, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return data['choices'][0]['text']
            else:
                error_text = await response.text()
                raise Exception(f"生成失败: {response.status} - {error_text}")
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """流式生成文本"""
        url = f"{self.base_url}/v1/completions"
        
        payload = {
            "model": self.model_path,
            "prompt": prompt,
            "max_tokens": kwargs.get('max_tokens', 2048),
            "temperature": kwargs.get('temperature', 0.8),
            "top_p": kwargs.get('top_p', 0.9),
            "frequency_penalty": kwargs.get('frequency_penalty', 0.0),
            "presence_penalty": kwargs.get('presence_penalty', 0.0),
            "stream": True
        }
        
        async with self.session.post(url, json=payload) as response:
            if response.status == 200:
                async for line in response.content:
                    if line:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            data_str = line[6:]
                            if data_str == '[DONE]':
                                break
                            try:
                                data = json.loads(data_str)
                                text = data['choices'][0].get('text', '')
                                if text:
                                    yield text
                            except json.JSONDecodeError:
                                continue
            else:
                error_text = await response.text()
                raise Exception(f"生成失败: {response.status} - {error_text}")
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """聊天接口"""
        url = f"{self.base_url}/v1/chat/completions"
        
        payload = {
            "model": self.model_path,
            "messages": messages,
            "max_tokens": kwargs.get('max_tokens', 2048),
            "temperature": kwargs.get('temperature', 0.8),
            "top_p": kwargs.get('top_p', 0.9),
            "frequency_penalty": kwargs.get('frequency_penalty', 0.0),
            "presence_penalty": kwargs.get('presence_penalty', 0.0),
            "stream": False
        }
        
        async with self.session.post(url, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return data['choices'][0]['message']['content']
            else:
                error_text = await response.text()
                raise Exception(f"聊天失败: {response.status} - {error_text}")
    
    async def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """批量生成"""
        tasks = []
        for prompt in prompts:
            task = self.generate(prompt, **kwargs)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"批量生成第{i}个失败: {result}")
                final_results.append("")
            else:
                final_results.append(result)
        
        return final_results
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_path": self.model_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "dtype": self.dtype,
            "server_url": self.base_url
        }


async def test_vllm_client():
    """测试vLLM客户端"""
    config = {
        "model_path": "/path/to/Qwen-72B-Chat",
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.9,
        "max_model_len": 8192,
        "dtype": "float16",
        "trust_remote_code": True,
        "port": 8000
    }
    
    async with VLLMClient(config) as client:
        # 测试生成
        prompt = "请解释IGZO TFT的工作原理："
        response = await client.generate(prompt, max_tokens=500)
        print(f"生成结果: {response}")
        
        # 测试聊天
        messages = [
            {"role": "system", "content": "你是一个半导体领域的专家。"},
            {"role": "user", "content": "IGZO的迁移率一般是多少？"}
        ]
        chat_response = await client.chat(messages)
        print(f"聊天结果: {chat_response}")
        
        # 测试批量生成
        prompts = [
            "什么是氧化物半导体？",
            "TFT的主要应用有哪些？",
            "如何提高IGZO的稳定性？"
        ]
        batch_results = await client.batch_generate(prompts, max_tokens=200)
        for i, result in enumerate(batch_results):
            print(f"批量生成{i+1}: {result[:100]}...")


if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_vllm_client())