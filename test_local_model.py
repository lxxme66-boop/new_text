#!/usr/bin/env python3
"""
测试本地模型配置
"""

import asyncio
import json
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LocalModels.ollama_client import OllamaClient


async def test_ollama_connection():
    """测试Ollama连接"""
    print("测试Ollama连接...")
    
    # Load config
    config_path = "config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    ollama_config = config['models']['local_models']['ollama']
    
    client = OllamaClient(
        base_url=ollama_config['base_url'],
        model_name=ollama_config['model_name'],
        timeout=ollama_config['timeout']
    )
    
    # Check connection
    if client.check_connection():
        print("✅ Ollama服务连接成功!")
        
        # Get available models
        models = client.get_available_models()
        print(f"\n可用模型: {models}")
        
        # Test generation
        print(f"\n使用模型 {ollama_config['model_name']} 进行测试...")
        
        test_prompt = "请解释什么是半导体？"
        response = await client.generate(test_prompt)
        
        if response:
            print(f"\n生成成功！")
            print(f"问题: {test_prompt}")
            print(f"回答: {response[:200]}...")
        else:
            print("❌ 生成失败")
            
    else:
        print("❌ 无法连接到Ollama服务")
        print("\n请确保:")
        print("1. Ollama已安装: curl -fsSL https://ollama.ai/install.sh | sh")
        print("2. Ollama服务正在运行: ollama serve")
        print("3. 已下载模型: ollama pull qwen:7b")


async def test_with_mock():
    """测试Mock模式"""
    print("\n\n测试Mock模式...")
    
    # Set mock mode
    os.environ['USE_MOCK_API'] = 'true'
    
    # Import and test
    from TextGeneration.Datageneration import input_text_process
    
    result = await input_text_process(
        "这是一段关于半导体技术的测试文本。",
        "test.txt",
        chunk_index=0,
        total_chunks=1,
        prompt_index=9
    )
    
    if result:
        print("✅ Mock模式测试成功!")
        print(f"结果: {json.dumps(result, ensure_ascii=False, indent=2)[:300]}...")
    else:
        print("❌ Mock模式测试失败")


async def main():
    """主函数"""
    print("=" * 60)
    print("本地模型配置测试")
    print("=" * 60)
    
    # Test Ollama
    await test_ollama_connection()
    
    # Test Mock mode
    await test_with_mock()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())