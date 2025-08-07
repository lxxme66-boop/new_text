#!/usr/bin/env python3
"""
智能文本QA生成系统 - 纯文本处理版
支持四种问题类型：事实型(15%)、比较型(15%)、推理型(50%)、开放型(20%)
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

# 设置Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入核心模块
try:
    from text_main_batch_inference_enhanced import main as retrieval_main
except ImportError:
    print("Warning: text_main_batch_inference_enhanced not available")
    retrieval_main = None

try:
    from clean_data import main as cleaning_main
except ImportError:
    print("Warning: clean_data not available")
    cleaning_main = None

from text_qa_generation_categorized import CategorizedQAGenerator, QuestionType
from TextQA.enhanced_quality_checker import TextQAQualityIntegrator

# 导入本地模型支持
try:
    from LocalModels.vllm_client import VLLMClient
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: VLLM not available")

# 设置日志
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline_text.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TextOnlyQAPipeline:
    """纯文本QA生成流水线"""
    
    def __init__(self, config_path: str = "config_local.json"):
        """初始化流水线"""
        self.config_path = config_path
        self.config = self.load_config()
        self.setup_directories()
        self.qa_generator = CategorizedQAGenerator(self.config)
        
        # 初始化统计信息
        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_duration': 0,
            'stages_completed': [],
            'total_files_processed': 0,
            'total_qa_pairs_generated': 0,
            'question_type_distribution': {},
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
            # 使用默认配置
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            "file_paths": {
                "input": {"text_dir": "data/texts"},
                "output": {
                    "base_dir": "data/output",
                    "retrieved_dir": "data/output/retrieved",
                    "qa_dir": "data/output/qa_results",
                    "quality_dir": "data/output/quality_checked",
                    "final_dir": "data/output/final"
                }
            },
            "processing": {
                "batch_size": 100,
                "max_concurrent": 20
            },
            "question_generation": {
                "batch_size": 50,
                "max_concurrent": 10
            }
        }
    
    def setup_directories(self):
        """创建必要的目录结构"""
        directories = [
            self.config['file_paths']['input']['text_dir'],
            self.config['file_paths']['output']['base_dir'],
            self.config['file_paths']['output']['retrieved_dir'],
            self.config['file_paths']['output']['qa_dir'],
            self.config['file_paths']['output']['quality_dir'],
            self.config['file_paths']['output']['final_dir'],
            'logs',
            'temp'
        ]
        
        for dir_path in directories:
            os.makedirs(dir_path, exist_ok=True)
        
        logger.info("目录结构创建完成")
    
    async def run_text_retrieval(self, input_path: str) -> str:
        """运行文本召回阶段"""
        logger.info("=" * 60)
        logger.info("开始文本召回阶段")
        logger.info("=" * 60)
        
        output_file = os.path.join(
            self.config['file_paths']['output']['retrieved_dir'],
            f"retrieved_{int(time.time())}.pkl"
        )
        
        if retrieval_main:
            try:
                # 调用文本召回模块
                await retrieval_main(
                    txt_path=input_path,
                    index=43,  # 默认索引
                    pool_size=self.config['processing']['batch_size'],
                    output_file=output_file
                )
                
                self.stats['stages_completed'].append('text_retrieval')
                logger.info(f"文本召回完成，结果保存到: {output_file}")
                return output_file
            except Exception as e:
                logger.error(f"文本召回失败: {e}")
                self.stats['errors'].append(f"text_retrieval: {str(e)}")
                # 返回输入路径，继续后续处理
                return input_path
        else:
            logger.warning("文本召回模块不可用，跳过此阶段")
            return input_path
    
    async def run_data_cleaning(self, input_file: str) -> str:
        """运行数据清理阶段"""
        logger.info("=" * 60)
        logger.info("开始数据清理阶段")
        logger.info("=" * 60)
        
        output_file = os.path.join(
            self.config['file_paths']['output']['cleaned_dir'],
            f"cleaned_{int(time.time())}.json"
        )
        
        if cleaning_main:
            try:
                # 调用数据清理模块
                cleaning_main(
                    input_file=input_file,
                    output_file=output_file
                )
                
                self.stats['stages_completed'].append('data_cleaning')
                logger.info(f"数据清理完成，结果保存到: {output_file}")
                return output_file
            except Exception as e:
                logger.error(f"数据清理失败: {e}")
                self.stats['errors'].append(f"data_cleaning: {str(e)}")
                return input_file
        else:
            logger.warning("数据清理模块不可用，跳过此阶段")
            return input_file
    
    async def run_qa_generation(self, input_file: str) -> str:
        """运行QA生成阶段（支持四种问题类型）"""
        logger.info("=" * 60)
        logger.info("开始QA生成阶段（四种问题类型）")
        logger.info("=" * 60)
        
        output_file = os.path.join(
            self.config['file_paths']['output']['qa_dir'],
            f"qa_categorized_{int(time.time())}.json"
        )
        
        try:
            # 读取输入数据
            if input_file.endswith('.json'):
                with open(input_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                # 处理其他格式
                data = self._load_input_data(input_file)
            
            # 提取内容
            contents = self._extract_contents(data)
            logger.info(f"提取到 {len(contents)} 条内容")
            
            # 生成分类的QA对
            qa_pairs = await self.qa_generator.generate_batch_categorized_qa(
                contents,
                batch_size=self.config['question_generation']['batch_size']
            )
            
            # 保存结果
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(qa_pairs, f, ensure_ascii=False, indent=4)
            
            # 更新统计信息
            self.stats['stages_completed'].append('qa_generation')
            self.stats['total_qa_pairs_generated'] = len(qa_pairs)
            
            # 统计问题类型分布
            stats = self.qa_generator.get_statistics(qa_pairs)
            self.stats['question_type_distribution'] = stats['type_ratios']
            
            logger.info(f"QA生成完成，生成 {len(qa_pairs)} 个问答对")
            logger.info(f"问题类型分布: {stats['type_ratios']}")
            
            # 保存统计信息
            stats_file = output_file.replace('.json', '_stats.json')
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=4)
            
            return output_file
            
        except Exception as e:
            logger.error(f"QA生成失败: {e}")
            self.stats['errors'].append(f"qa_generation: {str(e)}")
            raise
    
    def _load_input_data(self, input_file: str):
        """加载不同格式的输入数据"""
        if input_file.endswith('.pkl'):
            import pickle
            with open(input_file, 'rb') as f:
                return pickle.load(f)
        elif input_file.endswith('.txt'):
            with open(input_file, 'r', encoding='utf-8') as f:
                return [{"content": f.read()}]
        else:
            # 尝试作为目录处理
            if os.path.isdir(input_file):
                data = []
                for file_path in Path(input_file).glob('*.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data.append({"content": f.read(), "source": str(file_path)})
                return data
            else:
                raise ValueError(f"不支持的输入格式: {input_file}")
    
    def _extract_contents(self, data) -> List[str]:
        """从数据中提取内容"""
        contents = []
        
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    if 'content' in item:
                        contents.append(item['content'])
                    elif 'text' in item:
                        contents.append(item['text'])
                    elif 'text_content' in item:
                        contents.append(item['text_content'])
                elif isinstance(item, str):
                    contents.append(item)
        elif isinstance(data, dict):
            if 'content' in data:
                contents.append(data['content'])
        elif isinstance(data, str):
            contents.append(data)
        
        return contents
    
    async def run_quality_control(self, qa_file: str) -> str:
        """运行质量控制阶段"""
        logger.info("=" * 60)
        logger.info("开始质量控制阶段")
        logger.info("=" * 60)
        
        output_dir = self.config['file_paths']['output']['quality_dir']
        
        try:
            integrator = TextQAQualityIntegrator(self.config)
            quality_report = await integrator.enhanced_quality_check(
                qa_file_path=qa_file,
                output_dir=output_dir,
                quality_threshold=self.config.get('quality_control', {}).get('quality_threshold', 0.7)
            )
            
            self.stats['stages_completed'].append('quality_control')
            self.stats['quality_pass_rate'] = quality_report.get('pass_rate', 0)
            
            logger.info(f"质量控制完成，通过率: {self.stats['quality_pass_rate']:.2%}")
            
            # 返回高质量QA文件路径
            quality_file = os.path.join(output_dir, 'high_quality_qa.json')
            return quality_file
            
        except Exception as e:
            logger.error(f"质量控制失败: {e}")
            self.stats['errors'].append(f"quality_control: {str(e)}")
            return qa_file
    
    async def run_pipeline(self, input_path: str, stages: List[str] = None):
        """运行完整流水线"""
        self.stats['start_time'] = datetime.now()
        
        if stages is None:
            stages = ['retrieval', 'cleaning', 'qa_generation', 'quality_control']
        
        logger.info("=" * 60)
        logger.info("开始运行文本QA生成流水线")
        logger.info(f"输入路径: {input_path}")
        logger.info(f"执行阶段: {stages}")
        logger.info("=" * 60)
        
        current_file = input_path
        
        try:
            # 1. 文本召回
            if 'retrieval' in stages:
                current_file = await self.run_text_retrieval(current_file)
            
            # 2. 数据清理
            if 'cleaning' in stages:
                current_file = await self.run_data_cleaning(current_file)
            
            # 3. QA生成
            if 'qa_generation' in stages:
                current_file = await self.run_qa_generation(current_file)
            
            # 4. 质量控制
            if 'quality_control' in stages:
                current_file = await self.run_quality_control(current_file)
            
            # 完成统计
            self.stats['end_time'] = datetime.now()
            self.stats['total_duration'] = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
            
            # 保存最终报告
            self.save_final_report()
            
            logger.info("=" * 60)
            logger.info("流水线执行完成！")
            logger.info(f"总耗时: {self.stats['total_duration']:.2f} 秒")
            logger.info(f"生成QA对数: {self.stats['total_qa_pairs_generated']}")
            logger.info(f"质量通过率: {self.stats['quality_pass_rate']:.2%}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"流水线执行失败: {e}")
            self.stats['errors'].append(f"pipeline: {str(e)}")
            raise
    
    def save_final_report(self):
        """保存最终报告"""
        report_file = os.path.join(
            self.config['file_paths']['output']['final_dir'],
            f"pipeline_report_{int(time.time())}.json"
        )
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=4, default=str)
        
        logger.info(f"流水线报告保存到: {report_file}")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="智能文本QA生成系统 - 纯文本版")
    parser.add_argument("--config", type=str, default="config_local.json", help="配置文件路径")
    parser.add_argument("--input", type=str, default="data/texts", help="输入路径")
    parser.add_argument("--stages", type=str, nargs='+', 
                        choices=['retrieval', 'cleaning', 'qa_generation', 'quality_control'],
                        help="要执行的阶段")
    parser.add_argument("--all", action='store_true', help="执行所有阶段")
    
    args = parser.parse_args()
    
    # 创建流水线
    pipeline = TextOnlyQAPipeline(config_path=args.config)
    
    # 确定要执行的阶段
    if args.all or args.stages is None:
        stages = None  # 执行所有阶段
    else:
        stages = args.stages
    
    # 运行流水线
    await pipeline.run_pipeline(args.input, stages=stages)


if __name__ == "__main__":
    asyncio.run(main())