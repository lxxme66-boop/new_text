#!/usr/bin/env python3
"""
半导体QA生成系统 - 统一入口脚本
按照正确的流程调用各个模块：
1. text_processor.py - 文本预处理
2. text_main_batch_inference_enhanced.py - 文本召回与批量推理（新增）
3. clean_text_data.py - 数据清洗（新增）
4. semiconductor_qa_generator.py - 核心QA生成
5. 质量检查 - 独立的质量评估步骤（新增）
6. argument_data.py - 数据增强与重写
7. 最终输出整理 - 生成统计报告（新增）
"""

import asyncio
import argparse
import json
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pickle

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入各个模块
from text_processor import TextProcessor
from semiconductor_qa_generator import run_semiconductor_qa_generation

# 尝试导入argument_data模块
try:
    from argument_data import ArgumentDataProcessor
    ARGUMENT_DATA_AVAILABLE = True
except ImportError:
    ARGUMENT_DATA_AVAILABLE = False
    logger.warning("数据增强模块不可用（缺少volcenginesdkarkruntime）")
    
    # 创建一个mock类
    class ArgumentDataProcessor:
        """Mock ArgumentDataProcessor class"""
        def __init__(self):
            pass
        
        async def process_qa_data(self, *args, **kwargs):
            logger.warning("数据增强功能不可用，跳过此步骤")
            return args[0] if args else []

# 导入新增的模块
try:
    from text_main_batch_inference_enhanced import process_folders
    TEXT_RETRIEVAL_AVAILABLE = True
except ImportError:
    TEXT_RETRIEVAL_AVAILABLE = False
    logger.warning("文本召回模块不可用")

try:
    from clean_text_data import clean_data
    DATA_CLEANING_AVAILABLE = True
except ImportError:
    DATA_CLEANING_AVAILABLE = False
    logger.warning("数据清洗模块不可用")

try:
    from TextQA.enhanced_quality_checker import TextQAQualityIntegrator
    QUALITY_CHECK_AVAILABLE = True
except ImportError:
    QUALITY_CHECK_AVAILABLE = False
    logger.warning("增强质量检查模块不可用")


async def run_complete_pipeline(
    input_dir: str = "data/texts",
    output_dir: str = "data/qa_results",
    model_name: str = "qwq_32",
    batch_size: int = 32,
    gpu_devices: str = "4,5,6,7",
    enable_full_steps: bool = True  # 新增参数：是否启用完整7步骤
):
    """运行完整的QA生成流程"""
    
    logger.info("=== 开始半导体QA生成流程 ===")
    logger.info(f"模式: {'完整7步骤' if enable_full_steps else '精简3步骤'}")
    
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
    
    # 步骤2: 文本召回与批量推理（新增，可选）
    if enable_full_steps and TEXT_RETRIEVAL_AVAILABLE:
        logger.info("步骤2: 文本召回与批量推理")
        try:
            # 创建临时文件夹
            temp_folder = os.path.join(output_dir, "temp")
            os.makedirs(temp_folder, exist_ok=True)
            
            # 运行文本召回
            retrieval_results = await process_folders(
                folders=[input_dir],
                txt_path=input_dir,
                temporary_folder=temp_folder,
                index=43,  # 半导体领域的索引
                maximum_tasks=20,
                selected_task_number=500,
                storage_folder=output_dir,
                read_hist=False
            )
            
            # 保存召回结果
            retrieval_file = os.path.join(output_dir, "retrieval_results.pkl")
            with open(retrieval_file, 'wb') as f:
                pickle.dump(retrieval_results, f)
            
            logger.info(f"文本召回完成，生成 {len(retrieval_results)} 个召回结果")
        except Exception as e:
            logger.warning(f"文本召回步骤失败: {e}，继续执行后续步骤")
    
    # 步骤3: 数据清洗（新增，可选）
    if enable_full_steps and DATA_CLEANING_AVAILABLE:
        logger.info("步骤3: 数据清洗")
        try:
            # 如果有召回结果，使用召回结果；否则使用预处理结果
            input_for_cleaning = retrieval_file if 'retrieval_file' in locals() else preprocessed_file
            
            # 运行数据清洗
            cleaned_data = await clean_data(
                input_file=input_for_cleaning,
                output_dir=output_dir
            )
            
            # 保存清洗后的数据
            cleaned_file = os.path.join(output_dir, "cleaned_texts.json")
            with open(cleaned_file, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
            
            # 更新预处理文件路径
            preprocessed_file = cleaned_file
            logger.info(f"数据清洗完成，清洗后数据保存至: {cleaned_file}")
        except Exception as e:
            logger.warning(f"数据清洗步骤失败: {e}，使用原始预处理数据")
    
    # 步骤4: 使用semiconductor_qa_generator生成QA（原步骤2）
    logger.info(f"步骤{4 if enable_full_steps else 2}: 核心QA生成")
    
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
    
    # 步骤5: 独立的质量检查（新增，可选）
    if enable_full_steps and QUALITY_CHECK_AVAILABLE:
        logger.info("步骤5: 独立质量检查")
        try:
            # 加载配置
            config_path = "config.json"
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                config = {
                    "quality_control": {
                        "enhanced_quality_check": {
                            "quality_threshold": 0.7
                        }
                    }
                }
            
            # 初始化质量检查器
            quality_checker = TextQAQualityIntegrator(config)
            
            # 运行质量检查
            quality_report = await quality_checker.enhanced_quality_check(
                qa_file_path=qa_output_file,
                output_dir=output_dir,
                quality_threshold=config['quality_control']['enhanced_quality_check']['quality_threshold']
            )
            
            # 保存质量报告
            quality_report_file = os.path.join(output_dir, "quality_report.json")
            with open(quality_report_file, 'w', encoding='utf-8') as f:
                json.dump(quality_report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"质量检查完成，通过率: {quality_report.get('pass_rate', 0):.2%}")
            logger.info(f"质量报告保存至: {quality_report_file}")
        except Exception as e:
            logger.warning(f"质量检查步骤失败: {e}，继续执行后续步骤")
    
    # 步骤6: 数据增强与重写（原步骤3）
    logger.info(f"步骤{6 if enable_full_steps else 3}: 数据增强与重写")
    
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
    
    # 步骤7: 最终输出整理（新增）
    if enable_full_steps:
        logger.info("步骤7: 最终输出整理")
        
        # 生成详细统计信息
        stats = {
            "pipeline_info": {
                "mode": "完整7步骤" if enable_full_steps else "精简3步骤",
                "input_directory": input_dir,
                "output_directory": output_dir,
                "model_used": model_name,
                "gpu_devices": gpu_devices,
                "batch_size": batch_size
            },
            "processing_stats": {
                "total_texts_processed": len(processed_texts),
                "total_qa_generated": len(qa_results),
                "total_qa_enhanced": len(enhanced_data),
                "files_processed": len(text_files)
            }
        }
        
        # 添加质量检查统计（如果有）
        if 'quality_report' in locals():
            stats["quality_stats"] = {
                "total_qa_pairs": quality_report.get('total_qa_pairs', 0),
                "passed_qa_pairs": quality_report.get('passed_qa_pairs', 0),
                "pass_rate": quality_report.get('pass_rate', 0),
                "meets_threshold": quality_report.get('meets_threshold', False)
            }
        
        # 生成问题类型分布统计
        question_types = {}
        for item in enhanced_data:
            if 'question_type' in item:
                q_type = item['question_type']
                question_types[q_type] = question_types.get(q_type, 0) + 1
        
        stats["question_distribution"] = question_types
        
        # 保存统计信息
        stats_file = os.path.join(output_dir, "pipeline_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        # 生成摘要报告
        summary_report = f"""
=== 半导体QA生成流程完成 ===

处理模式: {stats['pipeline_info']['mode']}
输入目录: {stats['pipeline_info']['input_directory']}
输出目录: {stats['pipeline_info']['output_directory']}

处理统计:
- 处理文件数: {stats['processing_stats']['files_processed']}
- 处理文本段落数: {stats['processing_stats']['total_texts_processed']}
- 生成QA对数: {stats['processing_stats']['total_qa_generated']}
- 增强后QA对数: {stats['processing_stats']['total_qa_enhanced']}
"""
        
        if 'quality_stats' in stats:
            summary_report += f"""
质量检查:
- 通过率: {stats['quality_stats']['pass_rate']:.2%}
- 是否达标: {'是' if stats['quality_stats']['meets_threshold'] else '否'}
"""
        
        if question_types:
            summary_report += "\n问题类型分布:"
            for q_type, count in question_types.items():
                summary_report += f"\n- {q_type}: {count}"
        
        summary_report += f"""

输出文件:
- 最终数据集: {final_output_file}
- 统计信息: {stats_file}
"""
        
        # 保存摘要报告
        summary_file = os.path.join(output_dir, "summary_report.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_report)
        
        logger.info(summary_report)
        logger.info(f"摘要报告保存至: {summary_file}")
    else:
        # 精简模式下的简单统计
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
    parser.add_argument("--enable-full-steps", action="store_true",
                        help="启用完整7步骤流程（默认为精简3步骤）")
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径（可选，用于vLLM HTTP等配置）")
    
    args = parser.parse_args()
    
    # 如果提供了配置文件，加载并应用配置
    if args.config:
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"加载配置文件: {args.config}")
            
            # 设置环境变量以支持vLLM HTTP
            if config.get('api', {}).get('use_vllm_http'):
                os.environ['USE_VLLM_HTTP'] = 'true'
                os.environ['VLLM_SERVER_URL'] = config['api'].get('vllm_server_url', 'http://localhost:8000/v1')
                os.environ['USE_LOCAL_MODELS'] = str(config['api'].get('use_local_models', True)).lower()
                logger.info(f"启用vLLM HTTP模式，服务器地址: {os.environ['VLLM_SERVER_URL']}")
            
            # 从配置文件中获取处理参数（如果命令行没有指定，则使用配置文件的值）
            if args.batch_size == 32 and 'processing' in config:  # 使用默认值时才从配置文件读取
                args.batch_size = config['processing'].get('batch_size', args.batch_size)
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            sys.exit(1)
    
    # 运行异步流程
    asyncio.run(run_complete_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        gpu_devices=args.gpu_devices,
        enable_full_steps=args.enable_full_steps
    ))


if __name__ == "__main__":
    main()