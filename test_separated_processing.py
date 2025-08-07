#!/usr/bin/env python3
"""
测试脚本 - 演示分离的文本和PDF处理
"""

import asyncio
import os
import sys

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


async def test_text_processing():
    """测试文本处理"""
    print("\n" + "="*60)
    print("测试文本处理器")
    print("="*60)
    
    from text_processor import TextProcessor
    
    # 创建文本处理器
    processor = TextProcessor()
    
    # 检查文本目录
    text_dir = processor.text_dir
    print(f"\n文本目录: {text_dir}")
    
    if os.path.exists(text_dir):
        txt_files = [f for f in os.listdir(text_dir) if f.endswith('.txt')]
        print(f"找到 {len(txt_files)} 个txt文件:")
        for f in txt_files:
            print(f"  - {f}")
    else:
        print(f"错误: 文本目录不存在: {text_dir}")
        return
    
    # 处理一个示例文件
    if txt_files:
        sample_file = os.path.join(text_dir, txt_files[0])
        print(f"\n处理示例文件: {sample_file}")
        
        # 设置环境变量以使用mock模式（如果需要）
        os.environ['USE_MOCK_API'] = 'true'
        
        results = await processor.process_single_txt(sample_file)
        print(f"生成 {len(results)} 个结果")
        
        if results:
            print("\n第一个结果示例:")
            print(f"  源文件: {results[0].get('source_file', 'N/A')}")
            print(f"  块索引: {results[0].get('chunk_index', 'N/A')}")
            qa_pairs = results[0].get('qa_pairs', [])
            if qa_pairs:
                print(f"  QA对数量: {len(qa_pairs)}")


async def test_pdf_processing():
    """测试PDF处理"""
    print("\n" + "="*60)
    print("测试PDF处理器")
    print("="*60)
    
    from pdf_processor_main import PDFProcessorMain
    
    # 创建PDF处理器
    processor = PDFProcessorMain()
    
    # 检查PDF目录
    pdf_dir = processor.pdf_dir
    print(f"\nPDF目录: {pdf_dir}")
    
    if os.path.exists(pdf_dir):
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
        print(f"找到 {len(pdf_files)} 个PDF文件:")
        for f in pdf_files:
            print(f"  - {f}")
    else:
        print(f"注意: PDF目录不存在: {pdf_dir}")
        print("这是正常的，因为当前可能没有PDF文件")
    
    # 如果有PDF文件，处理一个示例
    if pdf_files:
        sample_file = os.path.join(pdf_dir, pdf_files[0])
        print(f"\n处理示例文件: {sample_file}")
        
        result = await processor.process_single_pdf(sample_file)
        if result['success']:
            print(f"处理成功!")
            print(f"  页数: {result['result']['total_pages']}")
            print(f"  图片数: {len(result['result']['images'])}")
        else:
            print(f"处理失败: {result['error']}")


async def main():
    """主函数"""
    print("\n智能文本QA生成系统 - 文本与PDF分离处理测试")
    print("=" * 80)
    
    # 测试文本处理
    await test_text_processing()
    
    # 测试PDF处理
    await test_pdf_processing()
    
    print("\n" + "="*80)
    print("测试完成！")
    print("\n总结:")
    print("1. 文本文件(.txt)现在从 data/texts 目录加载")
    print("2. PDF文件(.pdf)从 data/pdfs 目录加载")
    print("3. 文本处理结果保存在 data/qa_results/texts")
    print("4. PDF处理结果保存在 data/qa_results/pdfs")
    print("5. 两种处理完全分离，互不干扰")


if __name__ == "__main__":
    asyncio.run(main())