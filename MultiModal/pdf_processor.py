import os
import json
import fitz  # PyMuPDF
import base64
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """PDF处理器，用于提取文本和图像"""
    
    def __init__(self, output_dir: str = "output"):
        """
        初始化PDF处理器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def extract_text_and_images(self, pdf_path: str) -> Dict[str, Any]:
        """
        从PDF中提取文本和图像
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            提取结果，包含文本和图像信息
        """
        try:
            doc = fitz.open(pdf_path)
            pdf_name = Path(pdf_path).stem
            
            # 创建输出目录
            pdf_output_dir = os.path.join(self.output_dir, pdf_name)
            os.makedirs(pdf_output_dir, exist_ok=True)
            
            result = {
                'pdf_path': pdf_path,
                'pdf_name': pdf_name,
                'total_pages': len(doc),
                'pages': [],
                'images': [],
                'text_content': ""
            }
            
            all_text = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # 提取文本
                text = page.get_text()
                all_text.append(text)
                
                # 提取图像
                image_list = page.get_images(full=True)
                page_images = []
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        # 保存图像
                        img_name = f"page_{page_num+1}_img_{img_index+1}.png"
                        img_path = os.path.join(pdf_output_dir, img_name)
                        
                        if pix.n < 5:  # GRAY or RGB
                            pix.save(img_path)
                        else:  # CMYK
                            pix1 = fitz.Pixmap(fitz.csRGB, pix)
                            pix1.save(img_path)
                            pix1 = None
                        
                        pix = None
                        
                        # 记录图像信息
                        img_info = {
                            'page_num': page_num + 1,
                            'img_index': img_index + 1,
                            'img_name': img_name,
                            'img_path': img_path,
                            'relative_path': os.path.join(pdf_name, img_name)
                        }
                        
                        page_images.append(img_info)
                        result['images'].append(img_info)
                        
                    except Exception as e:
                        logger.warning(f"Error extracting image {img_index} from page {page_num}: {e}")
                
                # 记录页面信息
                page_info = {
                    'page_num': page_num + 1,
                    'text': text,
                    'images': page_images,
                    'image_count': len(page_images)
                }
                
                result['pages'].append(page_info)
            
            result['text_content'] = '\n'.join(all_text)
            doc.close()
            
            # 保存页面信息到JSON
            pages_json_path = os.path.join(pdf_output_dir, 'pages.json')
            with open(pages_json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'pages': list(range(1, len(doc) + 1)),
                    'total_pages': len(doc),
                    'images_count': len(result['images'])
                }, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Processed {pdf_path}: {len(doc)} pages, {len(result['images'])} images")
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return None
    
    def convert_to_markdown(self, pdf_result: Dict[str, Any]) -> str:
        """
        将PDF内容转换为Markdown格式
        
        Args:
            pdf_result: PDF处理结果
            
        Returns:
            Markdown格式的文本
        """
        if not pdf_result:
            return ""
        
        markdown_content = []
        markdown_content.append(f"# {pdf_result['pdf_name']}\n")
        
        for page_info in pdf_result['pages']:
            page_num = page_info['page_num']
            text = page_info['text']
            images = page_info['images']
            
            markdown_content.append(f"## Page {page_num}\n")
            
            # 添加文本内容
            if text.strip():
                markdown_content.append(text)
                markdown_content.append("\n")
            
            # 添加图像引用
            for img_info in images:
                img_path = img_info['relative_path']
                markdown_content.append(f"![Image](< image >) <!-- {img_path} -->\n")
        
        return '\n'.join(markdown_content)
    
    def extract_figure_references(self, text: str) -> List[str]:
        """
        从文本中提取图片引用（如Figure 1, Fig. 2等）
        
        Args:
            text: 文本内容
            
        Returns:
            图片引用列表
        """
        import re
        
        # 匹配各种图片引用格式
        patterns = [
            r'[Ff]igure\s+\d+',
            r'[Ff]ig\.?\s+\d+',
            r'图\s*\d+',
            r'图片\s*\d+',
            r'Figure\s+[A-Za-z]\d*',
            r'Fig\s+[A-Za-z]\d*'
        ]
        
        references = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            references.extend(matches)
        
        return list(set(references))


def process_pdf_folder(folder_path: str, output_dir: str = "output") -> List[Dict[str, Any]]:
    """
    批量处理文件夹中的PDF文件
    
    Args:
        folder_path: PDF文件夹路径
        output_dir: 输出目录
        
    Returns:
        处理结果列表
    """
    processor = PDFProcessor(output_dir)
    results = []
    
    # 查找所有PDF文件
    pdf_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_path in pdf_files:
        result = processor.extract_text_and_images(pdf_path)
        if result:
            results.append(result)
    
    # 保存总体统计信息
    summary = {
        'total_pdfs': len(pdf_files),
        'processed_pdfs': len(results),
        'total_pages': sum(r['total_pages'] for r in results),
        'total_images': sum(len(r['images']) for r in results)
    }
    
    summary_path = os.path.join(output_dir, 'processing_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Processing completed: {summary}")
    return results


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PDF处理工具")
    parser.add_argument("--input", required=True, help="输入PDF文件或文件夹路径")
    parser.add_argument("--output", default="output", help="输出目录")
    parser.add_argument("--markdown", action="store_true", help="生成Markdown格式输出")
    
    args = parser.parse_args()
    
    if os.path.isfile(args.input) and args.input.lower().endswith('.pdf'):
        # 处理单个PDF文件
        processor = PDFProcessor(args.output)
        result = processor.extract_text_and_images(args.input)
        
        if result and args.markdown:
            markdown_content = processor.convert_to_markdown(result)
            markdown_path = os.path.join(args.output, f"{result['pdf_name']}.md")
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            logger.info(f"Markdown saved to {markdown_path}")
    
    elif os.path.isdir(args.input):
        # 处理文件夹中的PDF文件
        results = process_pdf_folder(args.input, args.output)
        
        if args.markdown:
            processor = PDFProcessor(args.output)
            for result in results:
                markdown_content = processor.convert_to_markdown(result)
                markdown_path = os.path.join(args.output, f"{result['pdf_name']}.md")
                with open(markdown_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
            logger.info(f"Generated {len(results)} Markdown files")
    
    else:
        logger.error(f"Invalid input path: {args.input}")


if __name__ == "__main__":
    main()