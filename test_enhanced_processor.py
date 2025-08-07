#!/usr/bin/env python3
"""
测试增强文件处理器
验证PDF和文本文件的分离处理功能
"""

import asyncio
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from enhanced_file_processor import EnhancedFileProcessor


async def test_file_processor():
    """测试文件处理器的各种场景"""
    
    print("=" * 60)
    print("测试增强文件处理器")
    print("=" * 60)
    
    # 创建测试目录
    os.makedirs("data/pdfs", exist_ok=True)
    os.makedirs("data/texts", exist_ok=True)
    
    # 创建测试文件
    print("\n1. 创建测试文件...")
    
    # 创建测试文本文件
    test_txt_content = """这是一个测试文本文件。
用于验证文本文件处理功能。
包含多行内容。"""
    
    with open("data/texts/test_document.txt", "w", encoding="utf-8") as f:
        f.write(test_txt_content)
    
    # 创建另一个测试文件
    with open("data/texts/test_document2.txt", "w", encoding="utf-8") as f:
        f.write("第二个测试文档内容")
    
    print("✓ 测试文件创建完成")
    
    # 初始化处理器
    processor = EnhancedFileProcessor()
    
    # 测试场景1: 处理PDF目录
    print("\n2. 测试处理PDF目录...")
    pdf_results, txt_results = await processor.process_directory("data/pdfs")
    print(f"   PDF文件: {len(pdf_results)} 个")
    print(f"   文本文件: {len(txt_results)} 个")
    
    # 测试场景2: 处理文本目录
    print("\n3. 测试处理文本目录...")
    pdf_results, txt_results = await processor.process_directory("data/texts")
    print(f"   PDF文件: {len(pdf_results)} 个")
    print(f"   文本文件: {len(txt_results)} 个")
    
    # 显示处理结果
    if txt_results:
        print("\n   文本文件处理结果:")
        for result in txt_results:
            print(f"   - {result['file_name']}: {'成功' if result['success'] else '失败'}")
            if result['success']:
                print(f"     内容长度: {len(result['content'])} 字符")
            else:
                print(f"     错误: {result['error']}")
    
    # 测试场景3: 自动检测模式
    print("\n4. 测试自动检测模式...")
    pdf_results, txt_results = await processor.process_directory(".")
    print(f"   PDF文件: {len(pdf_results)} 个")
    print(f"   文本文件: {len(txt_results)} 个")
    
    # 测试场景4: 准备召回数据
    print("\n5. 测试准备召回数据...")
    retrieval_data = processor.prepare_for_retrieval(pdf_results, txt_results)
    print(f"   准备召回数据: {len(retrieval_data)} 条")
    
    if retrieval_data:
        print("\n   召回数据示例:")
        for i, data in enumerate(retrieval_data[:2]):  # 只显示前2条
            print(f"   [{i+1}] 文件: {data['source_file']}")
            print(f"       类型: {data['file_type']}")
            print(f"       内容预览: {data['content'][:50]}...")
    
    # 测试场景5: 测试文件类型检测
    print("\n6. 测试文件类型检测...")
    test_files = [
        "test.pdf",
        "document.txt",
        "readme.md",
        "data.json",
        "script.py"
    ]
    
    for file in test_files:
        file_type = processor.detect_file_type(file)
        print(f"   {file} -> {file_type}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    
    return retrieval_data


async def test_integration_with_pipeline():
    """测试与主流水线的集成"""
    
    print("\n\n测试与流水线集成...")
    print("=" * 60)
    
    # 模拟流水线调用
    from text_main_batch_inference_enhanced import main
    
    # 测试PDF目录
    print("\n1. 测试处理PDF目录...")
    try:
        results = await main(
            index=43,
            parallel_batch_size=10,
            pdf_path="data/pdfs",
            storage_folder="data/retrieved",
            selected_task_number=100,
            read_hist=False
        )
        print(f"   处理结果: {len(results)} 条数据")
    except Exception as e:
        print(f"   错误: {e}")
    
    # 测试文本目录
    print("\n2. 测试处理文本目录...")
    try:
        results = await main(
            index=43,
            parallel_batch_size=10,
            pdf_path="data/texts",
            storage_folder="data/retrieved",
            selected_task_number=100,
            read_hist=False
        )
        print(f"   处理结果: {len(results)} 条数据")
    except Exception as e:
        print(f"   错误: {e}")
    
    print("\n集成测试完成！")


if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_file_processor())
    
    # 如果需要测试集成
    # asyncio.run(test_integration_with_pipeline())