#!/usr/bin/env python3
"""
PDF处理器主程序 - 专门处理PDF文件（多模态处理）
"""

import asyncio
import argparse
import json
import os
import logging
from typing import List, Dict, Optional
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import PDF processing functions
try:
    from MultiModal.pdf_processor import PDFProcessor, process_pdf_folder
    logger.info("Successfully imported PDF processing functions")
except ImportError as e:
    logger.error(f"Failed to import PDF processing functions: {e}")
    raise


class PDFProcessorMain:
    """PDF处理器主类"""
    
    def __init__(self, config_path: str = "config.json"):
        """初始化PDF处理器"""
        self.config = self._load_config(config_path)
        self.pdf_dir = self.config.get('paths', {}).get('pdf_dir', 'data/pdfs')
        self.output_dir = self.config.get('paths', {}).get('pdf_output_dir', 'data/qa_results/pdfs')
        self.multimodal_config = self.config.get('multimodal', {})
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {}
    
    async def process_single_pdf(self, pdf_path: str) -> Dict:
        """处理单个PDF文件"""
        logger.info(f"Processing PDF file: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            logger.error(f"File not found: {pdf_path}")
            return {"success": False, "error": "File not found"}
        
        if not pdf_path.lower().endswith('.pdf'):
            logger.error(f"Not a PDF file: {pdf_path}")
            return {"success": False, "error": "Not a PDF file"}
        
        try:
            # 创建PDF处理器
            processor = PDFProcessor(self.output_dir)
            
            # 提取文本和图像
            result = processor.extract_text_and_images(pdf_path)
            
            if not result:
                logger.warning(f"No content extracted from {pdf_path}")
                return {"success": False, "error": "No content extracted"}
            
            # 处理提取的内容
            processed_result = await self._process_pdf_content(result)
            
            logger.info(f"Successfully processed {pdf_path}")
            return {
                "success": True,
                "file": pdf_path,
                "result": processed_result
            }
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _process_pdf_content(self, pdf_data: Dict) -> Dict:
        """处理PDF提取的内容"""
        # 这里可以添加更多的处理逻辑，比如：
        # - 对文本进行QA生成
        # - 对图像进行分析
        # - 生成多模态问答对
        
        processed_data = {
            "pdf_file": pdf_data.get("pdf_file"),
            "total_pages": pdf_data.get("total_pages"),
            "text_content": pdf_data.get("text_content", ""),
            "images": pdf_data.get("images", []),
            "metadata": pdf_data.get("metadata", {}),
            "processed_at": datetime.now().isoformat()
        }
        
        # 如果启用了多模态处理
        if self.multimodal_config.get('enabled', False):
            logger.info("Multimodal processing enabled")
            # 这里可以添加多模态处理逻辑
            processed_data['multimodal_analysis'] = {
                "text_image_pairs": [],
                "visual_qa_pairs": []
            }
        
        return processed_data
    
    async def process_pdf_folder(self, folder_path: Optional[str] = None,
                                max_files: Optional[int] = None) -> Dict:
        """处理文件夹中的所有PDF文件"""
        if folder_path is None:
            folder_path = self.pdf_dir
        
        logger.info(f"Processing PDF files from: {folder_path}")
        
        if not os.path.exists(folder_path):
            logger.error(f"Folder not found: {folder_path}")
            return {"success": False, "error": "Folder not found"}
        
        # 查找所有PDF文件
        pdf_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {folder_path}")
            return {"success": False, "error": "No PDF files found"}
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        # 限制处理文件数量
        if max_files:
            pdf_files = pdf_files[:max_files]
            logger.info(f"Processing only first {max_files} files")
        
        # 处理所有PDF文件
        all_results = []
        success_count = 0
        
        for pdf_path in pdf_files:
            result = await self.process_single_pdf(pdf_path)
            if result['success']:
                all_results.append(result['result'])
                success_count += 1
            else:
                logger.warning(f"Failed to process {pdf_path}: {result.get('error')}")
        
        # 保存结果
        output_file = os.path.join(self.output_dir, "pdf_processing_results.json")
        os.makedirs(self.output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to: {output_file}")
        
        # 生成统计信息
        stats = {
            "total_files": len(pdf_files),
            "processed_files": success_count,
            "failed_files": len(pdf_files) - success_count,
            "total_pages": sum(r.get('total_pages', 0) for r in all_results),
            "total_images": sum(len(r.get('images', [])) for r in all_results)
        }
        
        # 保存统计信息
        stats_file = os.path.join(self.output_dir, "pdf_processing_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        return {
            "success": True,
            "stats": stats,
            "output_file": output_file,
            "stats_file": stats_file
        }


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="PDF处理器 - 多模态处理PDF文件")
    parser.add_argument("--input", type=str, help="输入PDF文件或文件夹路径")
    parser.add_argument("--output", type=str, help="输出目录")
    parser.add_argument("--config", type=str, default="config.json", help="配置文件路径")
    parser.add_argument("--max-files", type=int, help="最大处理文件数")
    parser.add_argument("--enable-multimodal", action="store_true", help="启用多模态处理")
    
    args = parser.parse_args()
    
    # 创建处理器
    processor = PDFProcessorMain(args.config)
    
    # 设置输出目录
    if args.output:
        processor.output_dir = args.output
    
    # 启用多模态处理
    if args.enable_multimodal:
        processor.multimodal_config['enabled'] = True
    
    # 处理输入
    if args.input:
        if os.path.isfile(args.input) and args.input.lower().endswith('.pdf'):
            # 处理单个文件
            result = await processor.process_single_pdf(args.input)
            if result['success']:
                print(f"处理完成: {args.input}")
                print(f"页数: {result['result']['total_pages']}")
                print(f"图片数: {len(result['result']['images'])}")
            else:
                print(f"处理失败: {result['error']}")
        elif os.path.isdir(args.input):
            # 处理文件夹
            result = await processor.process_pdf_folder(
                args.input,
                max_files=args.max_files
            )
            if result['success']:
                print(f"处理完成:")
                print(f"  总文件数: {result['stats']['total_files']}")
                print(f"  成功处理: {result['stats']['processed_files']}")
                print(f"  失败文件: {result['stats']['failed_files']}")
                print(f"  总页数: {result['stats']['total_pages']}")
                print(f"  总图片数: {result['stats']['total_images']}")
                print(f"  结果保存在: {result['output_file']}")
            else:
                print(f"处理失败: {result['error']}")
        else:
            print(f"错误: 输入必须是PDF文件或文件夹")
    else:
        # 使用默认PDF目录
        result = await processor.process_pdf_folder(max_files=args.max_files)
        if result['success']:
            print(f"处理完成:")
            print(f"  总文件数: {result['stats']['total_files']}")
            print(f"  成功处理: {result['stats']['processed_files']}")
            print(f"  失败文件: {result['stats']['failed_files']}")
            print(f"  总页数: {result['stats']['total_pages']}")
            print(f"  总图片数: {result['stats']['total_images']}")
            print(f"  结果保存在: {result['output_file']}")
        else:
            print(f"处理失败: {result['error']}")


if __name__ == "__main__":
    asyncio.run(main())