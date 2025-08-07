#!/usr/bin/env python3
"""
测试文本召回功能
验证PDF和文本文件的分离处理
"""

import asyncio
import os
import sys
import json
import pickle

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_text_retrieval():
    """测试文本召回功能"""
    
    print("=" * 60)
    print("测试文本召回功能")
    print("=" * 60)
    
    # 确保目录存在
    os.makedirs("data/pdfs", exist_ok=True)
    os.makedirs("data/texts", exist_ok=True)
    os.makedirs("data/retrieved", exist_ok=True)
    
    # 检查文件
    print("\n1. 检查输入文件...")
    
    pdf_files = []
    if os.path.exists("data/pdfs"):
        pdf_files = [f for f in os.listdir("data/pdfs") if f.endswith('.pdf')]
        print(f"   PDF文件 ({len(pdf_files)}): {pdf_files}")
    
    txt_files = []
    if os.path.exists("data/texts"):
        txt_files = [f for f in os.listdir("data/texts") if f.endswith('.txt')]
        print(f"   文本文件 ({len(txt_files)}): {txt_files}")
    
    # 测试文本召回
    print("\n2. 测试文本召回模块...")
    
    try:
        from text_main_batch_inference_enhanced import main as retrieval_main
        
        # 测试处理文本文件
        if txt_files:
            print("\n   处理文本文件...")
            results = await retrieval_main(
                index=43,
                parallel_batch_size=10,
                pdf_path="data/texts",
                storage_folder="data/retrieved",
                selected_task_number=100,
                read_hist=False
            )
            print(f"   结果: {len(results)} 条数据")
            
            # 显示部分结果
            if results:
                print("\n   示例结果:")
                for i, result in enumerate(results[:2]):
                    print(f"   [{i+1}] {result.get('source_file', 'unknown')}")
        
        # 测试处理PDF文件
        if pdf_files:
            print("\n   处理PDF文件...")
            results = await retrieval_main(
                index=43,
                parallel_batch_size=10,
                pdf_path="data/pdfs",
                storage_folder="data/retrieved",
                selected_task_number=100,
                read_hist=False
            )
            print(f"   结果: {len(results)} 条数据")
    
    except Exception as e:
        print(f"   错误: {e}")
        import traceback
        traceback.print_exc()
    
    # 检查输出文件
    print("\n3. 检查输出文件...")
    output_file = "data/retrieved/total_response.pkl"
    if os.path.exists(output_file):
        print(f"   ✓ 输出文件存在: {output_file}")
        
        # 读取并显示内容
        try:
            with open(output_file, 'rb') as f:
                data = pickle.load(f)
                print(f"   包含 {len(data)} 条记录")
                
                if data:
                    print("\n   数据示例:")
                    for i, item in enumerate(data[:2]):
                        print(f"   [{i+1}] 类型: {type(item)}")
                        if isinstance(item, dict):
                            print(f"       键: {list(item.keys())}")
        except Exception as e:
            print(f"   读取失败: {e}")
    else:
        print(f"   ✗ 输出文件不存在")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


async def test_simple_processing():
    """简单的处理测试"""
    
    print("\n\n简单处理测试")
    print("=" * 60)
    
    # 直接测试文件处理
    from enhanced_file_processor import EnhancedFileProcessor
    
    processor = EnhancedFileProcessor()
    
    # 处理文本目录
    pdf_results, txt_results = await processor.process_directory("data/texts")
    
    print(f"处理结果: {len(pdf_results)} PDF, {len(txt_results)} TXT")
    
    # 准备数据
    all_data = processor.prepare_for_retrieval(pdf_results, txt_results)
    
    print(f"准备数据: {len(all_data)} 条")
    
    # 保存结果
    output_file = "data/retrieved/test_results.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(all_data, f)
    
    print(f"保存到: {output_file}")


if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_text_retrieval())
    
    # 运行简单测试
    asyncio.run(test_simple_processing())