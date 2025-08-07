#!/usr/bin/env python3
"""
增强文件处理器 - 智能识别和处理PDF/TXT文件
支持从不同目录加载不同类型的文件
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Union, Tuple
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    
try:
    import fitz  # pymupdf
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False
    
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

logger = logging.getLogger(__name__)


class EnhancedFileProcessor:
    """增强的文件处理器，支持PDF和TXT文件的智能处理"""
    
    def __init__(self, pdf_dir="data/pdfs", txt_dir="data/texts"):
        self.pdf_dir = pdf_dir
        self.txt_dir = txt_dir
        self.supported_extensions = {
            'pdf': ['.pdf'],
            'text': ['.txt', '.text', '.md']
        }
        
    def detect_file_type(self, file_path: str) -> str:
        """检测文件类型"""
        ext = Path(file_path).suffix.lower()
        
        for file_type, extensions in self.supported_extensions.items():
            if ext in extensions:
                return file_type
        
        return 'unknown'
    
    def get_files_by_type(self, directory: str, file_type: str) -> List[str]:
        """获取指定类型的文件列表"""
        if not os.path.exists(directory):
            logger.warning(f"目录不存在: {directory}")
            return []
        
        files = []
        extensions = self.supported_extensions.get(file_type, [])
        
        for file in os.listdir(directory):
            if any(file.lower().endswith(ext) for ext in extensions):
                files.append(os.path.join(directory, file))
        
        return files
    
    async def extract_text_from_pdf(self, pdf_path: str) -> str:
        """从PDF文件提取文本"""
        text = ""
        
        # 尝试使用pymupdf
        if FITZ_AVAILABLE:
            try:
                doc = fitz.open(pdf_path)
                for page in doc:
                    text += page.get_text()
                doc.close()
                
                if text.strip():
                    return text
            except Exception as e:
                logger.debug(f"pymupdf提取失败: {e}")
        
        # 尝试pdfplumber
        if PDFPLUMBER_AVAILABLE and not text.strip():
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                
                if text.strip():
                    return text
            except Exception as e:
                logger.debug(f"pdfplumber提取失败: {e}")
        
        # 最后尝试PyPDF2
        if PYPDF2_AVAILABLE and not text.strip():
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        text += page.extract_text()
                    return text
            except Exception as e:
                logger.debug(f"PyPDF2提取失败: {e}")
        
        # 如果没有可用的PDF库
        if not any([FITZ_AVAILABLE, PDFPLUMBER_AVAILABLE, PYPDF2_AVAILABLE]):
            logger.error(f"没有可用的PDF处理库，无法提取 {pdf_path}")
            return ""
        
        logger.error(f"所有PDF提取方法失败 {pdf_path}")
        return ""
    
    async def read_text_file(self, txt_path: str) -> str:
        """读取文本文件"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # 尝试其他编码
            try:
                with open(txt_path, 'r', encoding='gbk') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"文本文件读取失败 {txt_path}: {e}")
                return ""
    
    async def process_file(self, file_path: str) -> Dict:
        """处理单个文件"""
        file_type = self.detect_file_type(file_path)
        
        result = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'file_type': file_type,
            'content': '',
            'success': False,
            'error': None
        }
        
        try:
            if file_type == 'pdf':
                result['content'] = await self.extract_text_from_pdf(file_path)
            elif file_type == 'text':
                result['content'] = await self.read_text_file(file_path)
            else:
                result['error'] = f"不支持的文件类型: {file_type}"
                return result
            
            result['success'] = bool(result['content'].strip())
            if not result['success']:
                result['error'] = "文件内容为空"
                
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"处理文件失败 {file_path}: {e}")
        
        return result
    
    async def process_directory(self, input_path: str) -> Tuple[List[Dict], List[Dict]]:
        """
        处理目录，智能识别PDF和文本文件
        返回: (pdf_results, txt_results)
        """
        pdf_results = []
        txt_results = []
        
        # 判断输入路径
        if "pdf" in input_path.lower() or input_path == self.pdf_dir:
            # 处理PDF目录
            pdf_files = self.get_files_by_type(input_path, 'pdf')
            logger.info(f"找到 {len(pdf_files)} 个PDF文件")
            
            for pdf_file in pdf_files:
                result = await self.process_file(pdf_file)
                pdf_results.append(result)
            
            # 同时检查文本目录
            if os.path.exists(self.txt_dir):
                txt_files = self.get_files_by_type(self.txt_dir, 'text')
                logger.info(f"在 {self.txt_dir} 找到 {len(txt_files)} 个文本文件")
                
                for txt_file in txt_files:
                    result = await self.process_file(txt_file)
                    txt_results.append(result)
                    
        elif "text" in input_path.lower() or input_path == self.txt_dir:
            # 处理文本目录
            txt_files = self.get_files_by_type(input_path, 'text')
            logger.info(f"找到 {len(txt_files)} 个文本文件")
            
            for txt_file in txt_files:
                result = await self.process_file(txt_file)
                txt_results.append(result)
            
            # 同时检查PDF目录
            if os.path.exists(self.pdf_dir):
                pdf_files = self.get_files_by_type(self.pdf_dir, 'pdf')
                logger.info(f"在 {self.pdf_dir} 找到 {len(pdf_files)} 个PDF文件")
                
                for pdf_file in pdf_files:
                    result = await self.process_file(pdf_file)
                    pdf_results.append(result)
                    
        else:
            # 自动检测并处理两种类型
            logger.info(f"自动检测模式: {input_path}")
            
            # 检查PDF目录
            if os.path.exists(self.pdf_dir):
                pdf_files = self.get_files_by_type(self.pdf_dir, 'pdf')
                logger.info(f"在 {self.pdf_dir} 找到 {len(pdf_files)} 个PDF文件")
                
                for pdf_file in pdf_files:
                    result = await self.process_file(pdf_file)
                    pdf_results.append(result)
            
            # 检查文本目录
            if os.path.exists(self.txt_dir):
                txt_files = self.get_files_by_type(self.txt_dir, 'text')
                logger.info(f"在 {self.txt_dir} 找到 {len(txt_files)} 个文本文件")
                
                for txt_file in txt_files:
                    result = await self.process_file(txt_file)
                    txt_results.append(result)
        
        return pdf_results, txt_results
    
    def prepare_for_retrieval(self, pdf_results: List[Dict], txt_results: List[Dict]) -> List[Dict]:
        """准备用于文本召回的数据格式"""
        all_results = []
        
        # 处理PDF结果
        for result in pdf_results:
            if result['success']:
                all_results.append({
                    'source_file': result['file_name'],
                    'file_type': 'pdf',
                    'content': result['content'],
                    'metadata': {
                        'file_path': result['file_path'],
                        'extraction_method': 'pdf_extraction'
                    }
                })
        
        # 处理文本结果
        for result in txt_results:
            if result['success']:
                all_results.append({
                    'source_file': result['file_name'],
                    'file_type': 'text',
                    'content': result['content'],
                    'metadata': {
                        'file_path': result['file_path'],
                        'extraction_method': 'direct_read'
                    }
                })
        
        return all_results


async def test_processor():
    """测试文件处理器"""
    processor = EnhancedFileProcessor()
    
    # 创建测试目录
    os.makedirs("data/pdfs", exist_ok=True)
    os.makedirs("data/texts", exist_ok=True)
    
    # 测试处理
    pdf_results, txt_results = await processor.process_directory("data/pdfs")
    
    print(f"PDF处理结果: {len(pdf_results)} 个文件")
    print(f"文本处理结果: {len(txt_results)} 个文件")
    
    # 准备召回数据
    retrieval_data = processor.prepare_for_retrieval(pdf_results, txt_results)
    print(f"准备召回数据: {len(retrieval_data)} 条")
    
    return retrieval_data


if __name__ == "__main__":
    asyncio.run(test_processor())