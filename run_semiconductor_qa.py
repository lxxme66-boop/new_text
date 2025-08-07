#!/usr/bin/env python3
"""
半导体QA生成系统 - 统一入口脚本
按照正确的流程调用各个模块：
1. text_processor.py - 文本预处理
2. semiconductor_qa_generator.py - 核心QA生成
3. argument_data.py - 数据增强与重写
"""

import asyncio
import argparse
import json
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入各个模块
from text_processor import TextProcessor
from semiconductor_qa_generator import run_semiconductor_qa_generation
from argument_data import ArgumentDataProcessor


async def run_complete_pipeline(
    input_dir: str = "data/texts",
    output_dir: str = "data/qa_results",
    model_name: str = "qwq_32",
    batch_size: int = 32,
    gpu_devices: str = "4,5,6,7"
):
    """运行完整的QA生成流程"""
    
    logger.info("=== 开始半导体QA生成流程 ===")
    
    # 步骤1: 文本预处理
    logger.info("步骤1: 文本预处理")
    text_processor = TextProcessor()
    
    # 处理所有文本文件
    processed_texts = []
    text_files = list(Path(input_dir).glob("*.txt"))
    
    for txt_file in text_files:
        logger.info(f"处理文本文件: {txt_file}")
        try:
            results = await text_processor.process_single_txt(str(txt_file))
            processed_texts.extend(results)
        except Exception as e:
            logger.error(f"处理文件 {txt_file} 时出错: {e}")
    
    # 保存预处理结果
    preprocessed_file = os.path.join(output_dir, "preprocessed_texts.json")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(preprocessed_file, 'w', encoding='utf-8') as f:
        json.dump(processed_texts, f, ensure_ascii=False, indent=2)
    
    logger.info(f"预处理完成，共处理 {len(processed_texts)} 个文本段落")
    
    # 步骤2: 使用semiconductor_qa_generator生成QA
    logger.info("步骤2: 核心QA生成")
    
    # 准备输入文件列表
    input_files = [preprocessed_file]
    qa_output_file = os.path.join(output_dir, "qa_generated.json")
    output_files = [qa_output_file]
    
    # 运行QA生成
    qa_results = await run_semiconductor_qa_generation(
        raw_folders=input_files,
        save_paths=output_files,
        model_name=model_name,
        batch_size=batch_size,
        gpu_devices=gpu_devices
    )
    
    logger.info(f"QA生成完成，生成了 {len(qa_results)} 个QA对")
    
    # 步骤3: 数据增强与重写
    logger.info("步骤3: 数据增强与重写")
    
    # 初始化数据增强处理器
    argument_processor = ArgumentDataProcessor()
    
    # 加载生成的QA数据
    with open(qa_output_file, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    # 进行数据增强
    enhanced_data = await argument_processor.enhance_qa_data(qa_data)
    
    # 保存最终结果
    final_output_file = os.path.join(output_dir, "final_qa_dataset.json")
    with open(final_output_file, 'w', encoding='utf-8') as f:
        json.dump(enhanced_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"数据增强完成，最终数据集保存至: {final_output_file}")
    
    # 生成统计信息
    stats = {
        "total_texts_processed": len(processed_texts),
        "total_qa_generated": len(qa_results),
        "total_qa_enhanced": len(enhanced_data),
        "input_directory": input_dir,
        "output_directory": output_dir,
        "model_used": model_name
    }
    
    stats_file = os.path.join(output_dir, "pipeline_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    logger.info("=== QA生成流程完成 ===")
    logger.info(f"统计信息已保存至: {stats_file}")
    
    return enhanced_data


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="半导体QA生成系统")
    parser.add_argument("--input-dir", type=str, default="data/texts",
                        help="输入文本文件目录")
    parser.add_argument("--output-dir", type=str, default="data/qa_results",
                        help="输出结果目录")
    parser.add_argument("--model", type=str, default="qwq_32",
                        help="使用的模型名称")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="批处理大小")
    parser.add_argument("--gpu-devices", type=str, default="4,5,6,7",
                        help="GPU设备ID")
    
    args = parser.parse_args()
    
    # 运行异步流程
    asyncio.run(run_complete_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        gpu_devices=args.gpu_devices
    ))


if __name__ == "__main__":
    main()