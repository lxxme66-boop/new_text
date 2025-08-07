#!/usr/bin/env python3
"""
智能文本QA生成系统 - 整合版统一流水线
整合了数据召回、清理、QA生成、质量控制、多模态处理和本地模型支持的完整流水线

功能模块：
1. 文本召回与检索 (Text Retrieval)
2. 数据清理与预处理 (Data Cleaning)  
3. 智能QA生成 (QA Generation)
4. 增强质量控制 (Quality Control)
5. 多模态处理 (Multimodal Processing)
6. 本地模型支持 (Local Models)
"""

import asyncio
import argparse
import json
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import time
from datetime import datetime

# 导入核心模块
from text_main_batch_inference_enhanced import main as retrieval_main
from clean_data import main as cleaning_main
from text_qa_generation import main as qa_generation_main
from TextQA.enhanced_quality_checker import TextQAQualityIntegrator

# 导入多模态和本地模型支持
try:
    from MultiModal.pdf_processor import PDFProcessor
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False
    print("Warning: MultiModal processing not available")

try:
    from LocalModels.ollama_client import OllamaClient
    LOCAL_MODELS_AVAILABLE = True
except ImportError:
    LOCAL_MODELS_AVAILABLE = False
    print("Warning: Local models not available")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class IntegratedQAPipeline:
    """整合版QA生成流水线"""
    
    def __init__(self, config_path: str = "config.json"):
        """初始化流水线"""
        self.config_path = config_path
        self.config = self.load_config()
        self.setup_directories()
        
        # 初始化统计信息
        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_duration': 0,
            'stages_completed': [],
            'total_files_processed': 0,
            'total_qa_pairs_generated': 0,
            'quality_pass_rate': 0.0,
            'errors': []
        }
    
    def load_config(self) -> Dict:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"配置文件加载成功: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            raise
    
    def setup_directories(self):
        """创建必要的目录结构"""
        directories = [
            self.config['file_paths']['input']['base_dir'],
            self.config['file_paths']['output']['base_dir'],
            self.config['file_paths']['output']['retrieved_dir'],
            self.config['file_paths']['output']['cleaned_dir'],
            self.config['file_paths']['output']['qa_dir'],
            self.config['file_paths']['output']['quality_dir'],
            self.config['file_paths']['output']['final_dir'],
            self.config['file_paths']['temp']['base_dir'],
            self.config['file_paths']['temp']['cache_dir'],
            self.config['file_paths']['temp']['logs_dir']
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        logger.info("目录结构创建完成")
    
    async def run_full_pipeline(self, input_path: str, domain: str = "semiconductor") -> Dict:
        """运行完整流水线"""
        logger.info("=" * 60)
        logger.info("开始运行智能文本QA生成系统 - 整合版")
        logger.info("=" * 60)
        
        self.stats['start_time'] = datetime.now()
        
        try:
            # 阶段1: 文本召回
            if os.path.isdir(input_path):
                retrieval_results = await self.stage_text_retrieval(input_path, domain)
                self.stats['stages_completed'].append('text_retrieval')
            else:
                logger.info("跳过文本召回阶段 - 输入为单个文件")
                retrieval_results = {'output_file': input_path}
            
            # 阶段2: 数据清理
            cleaning_results = await self.stage_data_cleaning(
                retrieval_results.get('output_file', input_path)
            )
            self.stats['stages_completed'].append('data_cleaning')
            
            # 阶段3: QA生成
            qa_results = await self.stage_qa_generation(
                cleaning_results['output_file'], domain
            )
            self.stats['stages_completed'].append('qa_generation')
            
            # 阶段4: 质量控制
            quality_results = await self.stage_quality_control(
                qa_results['output_file']
            )
            self.stats['stages_completed'].append('quality_control')
            
            # 阶段5: 最终整理
            final_results = await self.stage_final_processing(
                quality_results['output_file'], domain
            )
            self.stats['stages_completed'].append('final_processing')
            
            # 生成最终报告
            report = self.generate_final_report(final_results)
            
            self.stats['end_time'] = datetime.now()
            self.stats['total_duration'] = (
                self.stats['end_time'] - self.stats['start_time']
            ).total_seconds()
            
            logger.info("=" * 60)
            logger.info("流水线执行完成")
            logger.info(f"总耗时: {self.stats['total_duration']:.2f} 秒")
            logger.info("=" * 60)
            
            return {
                'success': True,
                'final_output': final_results,
                'report': report,
                'stats': self.stats
            }
            
        except Exception as e:
            logger.error(f"流水线执行失败: {e}")
            self.stats['errors'].append(str(e))
            return {
                'success': False,
                'error': str(e),
                'stats': self.stats
            }
    
    async def stage_text_retrieval(self, input_path: str, domain: str) -> Dict:
        """阶段1: 文本召回"""
        logger.info("阶段1: 开始文本召回...")
        
        output_dir = self.config['file_paths']['output']['retrieved_dir']
        
        # 智能判断输入路径类型
        actual_input_path = input_path
        
        # 如果输入路径是data/pdfs但不存在，检查data/texts
        if input_path == "data/pdfs" and not os.path.exists(input_path):
            if os.path.exists("data/texts"):
                logger.info("PDF目录不存在，切换到文本目录: data/texts")
                actual_input_path = "data/texts"
            else:
                # 创建pdfs目录
                os.makedirs("data/pdfs", exist_ok=True)
                logger.warning(f"创建PDF目录: {input_path}")
        
        # 如果路径包含pdf但目录为空，尝试texts目录
        if "pdf" in input_path.lower():
            pdf_files = []
            if os.path.exists(input_path):
                pdf_files = [f for f in os.listdir(input_path) if f.endswith('.pdf')]
            
            if not pdf_files and os.path.exists("data/texts"):
                txt_files = [f for f in os.listdir("data/texts") if f.endswith('.txt')]
                if txt_files:
                    logger.info(f"PDF目录为空，发现文本文件在data/texts，切换处理")
                    actual_input_path = "data/texts"
        
        # 构建召回参数
        retrieval_args = {
            'index': self.config['data_retrieval']['retrieval_indices'].get(domain, 43),
            'parallel_batch_size': self.config['processing']['parallel_batch_size'],
            'pdf_path': actual_input_path,
            'storage_folder': output_dir,
            'selected_task_number': self.config['processing']['selected_task_number'],
            'read_hist': False
        }
        
        logger.info(f"实际处理路径: {actual_input_path}")
        
        try:
            # 调用召回模块
            await retrieval_main(
                index=retrieval_args['index'],
                parallel_batch_size=retrieval_args['parallel_batch_size'],
                pdf_path=retrieval_args['pdf_path'],
                storage_folder=retrieval_args['storage_folder'],
                selected_task_number=retrieval_args['selected_task_number'],
                read_hist=retrieval_args['read_hist']
            )
            
            output_file = os.path.join(output_dir, 'total_response.pkl')
            logger.info(f"文本召回完成，输出: {output_file}")
            
            return {
                'success': True,
                'output_file': output_file,
                'stage': 'text_retrieval'
            }
        except Exception as e:
            import traceback
            logger.error(f"文本召回失败: {e}")
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            raise
    
    async def stage_data_cleaning(self, input_file: str) -> Dict:
        """阶段2: 数据清理"""
        logger.info("阶段2: 开始数据清理...")
        
        output_dir = self.config['file_paths']['output']['cleaned_dir']
        
        try:
            # 调用清理模块
            cleaned_file = await cleaning_main(
                input_file=input_file,
                output_dir=output_dir,
                copy_parsed_pdf=False
            )
            
            logger.info(f"数据清理完成，输出: {cleaned_file}")
            
            return {
                'success': True,
                'output_file': cleaned_file,
                'stage': 'data_cleaning'
            }
        except Exception as e:
            logger.error(f"数据清理失败: {e}")
            raise
    
    async def stage_qa_generation(self, input_file: str, domain: str) -> Dict:
        """阶段3: QA生成"""
        logger.info("阶段3: 开始QA生成...")
        
        output_dir = self.config['file_paths']['output']['qa_dir']
        
        # 获取领域特定的prompt
        prompt_index = self.config['professional_domains']['domain_specific_prompts'].get(
            domain, [343]
        )[0]
        
        try:
            # 构建QA生成参数
            qa_args = {
                'index': prompt_index,
                'file_path': input_file,
                'pool_size': self.config['processing']['pool_size'],
                'output_file': output_dir,
                'ark_url': self.config['api']['ark_url'],
                'api_key': self.config['api']['api_key'],
                'model': self.config['models']['qa_generator_model']['path'],
                'check_task': False,
                'user_stream': False,
                'enhanced_quality': False
            }
            
            # 调用QA生成模块
            results = await qa_generation_main(**qa_args)
            
            output_file = os.path.join(output_dir, f"results_{prompt_index}.json")
            self.stats['total_qa_pairs_generated'] = len(results) if results else 0
            
            logger.info(f"QA生成完成，生成 {self.stats['total_qa_pairs_generated']} 个QA对")
            logger.info(f"输出文件: {output_file}")
            
            return {
                'success': True,
                'output_file': output_file,
                'qa_count': self.stats['total_qa_pairs_generated'],
                'stage': 'qa_generation'
            }
        except Exception as e:
            logger.error(f"QA生成失败: {e}")
            raise
    
    async def stage_quality_control(self, input_file: str) -> Dict:
        """阶段4: 质量控制"""
        logger.info("阶段4: 开始质量控制...")
        
        output_dir = self.config['file_paths']['output']['quality_dir']
        
        try:
            # 使用增强质量检查
            integrator = TextQAQualityIntegrator(self.config)
            
            quality_report = await integrator.enhanced_quality_check(
                qa_file_path=input_file,
                output_dir=output_dir,
                quality_threshold=self.config['quality_control']['enhanced_quality_check']['quality_threshold']
            )
            
            self.stats['quality_pass_rate'] = quality_report.get('pass_rate', 0.0)
            
            logger.info("质量控制完成")
            logger.info(f"质量通过率: {self.stats['quality_pass_rate']:.2%}")
            
            output_file = os.path.join(output_dir, "quality_checked_qa.json")
            
            return {
                'success': True,
                'output_file': output_file,
                'quality_report': quality_report,
                'stage': 'quality_control'
            }
        except Exception as e:
            logger.error(f"质量控制失败: {e}")
            raise
    
    async def stage_final_processing(self, input_file: str, domain: str) -> Dict:
        """阶段5: 最终处理"""
        logger.info("阶段5: 开始最终处理...")
        
        final_dir = self.config['file_paths']['output']['final_dir']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # 读取质量检查后的数据
            with open(input_file, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
            
            # 添加元数据
            final_data = {
                'metadata': {
                    'system_name': self.config['system_info']['name'],
                    'version': self.config['system_info']['version'],
                    'domain': domain,
                    'generation_time': timestamp,
                    'total_qa_pairs': len(qa_data) if isinstance(qa_data, list) else qa_data.get('total_qa_pairs', 0),
                    'quality_threshold': self.config['quality_control']['enhanced_quality_check']['quality_threshold'],
                    'quality_pass_rate': self.stats['quality_pass_rate']
                },
                'qa_pairs': qa_data,
                'pipeline_stats': self.stats
            }
            
            # 保存最终结果
            final_output_file = os.path.join(
                final_dir, 
                f"final_qa_results_{domain}_{timestamp}.json"
            )
            
            with open(final_output_file, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, ensure_ascii=False, indent=4)
            
            logger.info(f"最终处理完成，输出: {final_output_file}")
            
            return {
                'success': True,
                'output_file': final_output_file,
                'final_data': final_data,
                'stage': 'final_processing'
            }
        except Exception as e:
            logger.error(f"最终处理失败: {e}")
            raise
    
    def generate_final_report(self, final_results: Dict) -> Dict:
        """生成最终报告"""
        report = {
            'pipeline_summary': {
                'system_name': self.config['system_info']['name'],
                'version': self.config['system_info']['version'],
                'execution_time': self.stats['total_duration'],
                'stages_completed': self.stats['stages_completed'],
                'success': len(self.stats['errors']) == 0
            },
            'processing_stats': {
                'total_files_processed': self.stats['total_files_processed'],
                'total_qa_pairs_generated': self.stats['total_qa_pairs_generated'],
                'quality_pass_rate': self.stats['quality_pass_rate']
            },
            'output_files': {
                'final_output': final_results.get('output_file'),
                'logs': 'logs/pipeline.log'
            },
            'errors': self.stats['errors']
        }
        
        # 保存报告
        report_file = os.path.join(
            self.config['file_paths']['output']['final_dir'],
            f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
        
        logger.info(f"流水线报告已保存: {report_file}")
        
        return report

    async def run_single_stage(self, stage: str, input_path: str, **kwargs) -> Dict:
        """运行单个阶段"""
        logger.info(f"运行单个阶段: {stage}")
        
        if stage == "text_retrieval":
            return await self.stage_text_retrieval(input_path, kwargs.get('domain', 'semiconductor'))
        elif stage == "data_cleaning":
            return await self.stage_data_cleaning(input_path)
        elif stage == "qa_generation":
            return await self.stage_qa_generation(input_path, kwargs.get('domain', 'semiconductor'))
        elif stage == "quality_control":
            return await self.stage_quality_control(input_path)
        elif stage == "final_processing":
            return await self.stage_final_processing(input_path, kwargs.get('domain', 'semiconductor'))
        else:
            raise ValueError(f"未知的阶段: {stage}")

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="智能文本QA生成系统 - 整合版")
    
    # 基本参数
    parser.add_argument("--mode", type=str, default="full_pipeline", 
                       choices=["full_pipeline", "text_retrieval", "data_cleaning", 
                               "qa_generation", "quality_control", "final_processing"],
                       help="运行模式")
    parser.add_argument("--input_path", type=str, required=True, help="输入路径")
    parser.add_argument("--output_path", type=str, help="输出路径（可选）")
    parser.add_argument("--config", type=str, default="config.json", help="配置文件路径")
    parser.add_argument("--domain", type=str, default="semiconductor", 
                       choices=["semiconductor", "optics", "materials", "physics", "chemistry"],
                       help="专业领域")
    
    # 高级参数
    parser.add_argument("--batch_size", type=int, help="批处理大小")
    parser.add_argument("--pool_size", type=int, help="并发池大小")
    parser.add_argument("--quality_threshold", type=float, help="质量阈值")
    parser.add_argument("--use_local_models", action="store_true", help="使用本地模型")
    parser.add_argument("--enable_multimodal", action="store_true", help="启用多模态处理")
    
    args = parser.parse_args()
    
    # 初始化流水线
    pipeline = IntegratedQAPipeline(args.config)
    
    # 更新配置（如果提供了参数）
    if args.batch_size:
        pipeline.config['processing']['batch_size'] = args.batch_size
    if args.pool_size:
        pipeline.config['processing']['pool_size'] = args.pool_size
    if args.quality_threshold:
        pipeline.config['quality_control']['enhanced_quality_check']['quality_threshold'] = args.quality_threshold
    if args.use_local_models:
        pipeline.config['api']['use_local_models'] = True
    if args.enable_multimodal:
        pipeline.config['multimodal']['enabled'] = True
    
    try:
        if args.mode == "full_pipeline":
            # 运行完整流水线
            results = await pipeline.run_full_pipeline(args.input_path, args.domain)
        else:
            # 运行单个阶段
            results = await pipeline.run_single_stage(
                args.mode, args.input_path, domain=args.domain
            )
        
        if results['success']:
            print("\n" + "=" * 60)
            print("✅ 流水线执行成功！")
            if 'final_output' in results:
                print(f"📁 最终输出: {results['final_output']['output_file']}")
            if 'report' in results:
                print(f"📊 处理统计: 生成 {results['stats']['total_qa_pairs_generated']} 个QA对")
                print(f"⭐ 质量通过率: {results['stats']['quality_pass_rate']:.2%}")
            print("=" * 60)
        else:
            print(f"\n❌ 流水线执行失败: {results['error']}")
            return 1
            
    except Exception as e:
        logger.error(f"程序执行异常: {e}")
        print(f"\n💥 程序执行异常: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))