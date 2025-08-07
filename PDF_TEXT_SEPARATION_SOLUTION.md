# PDF和文本文件分离处理解决方案

## 问题分析

原始问题：
- 系统期望从 `data/pdfs` 加载PDF文件，但实际上只找到了文本文件
- 代码只查找 `.txt` 文件，导致无法正确处理PDF文件
- 需要实现PDF和文本文件的分离处理

## 解决方案

### 1. 增强文件处理器 (enhanced_file_processor.py)

创建了一个智能文件处理器，能够：
- 自动检测文件类型（PDF/文本）
- 从不同目录加载不同类型的文件
- 支持多种PDF提取库（PyPDF2、pymupdf、pdfplumber）
- 统一输出格式供后续处理

```python
class EnhancedFileProcessor:
    def __init__(self, pdf_dir="data/pdfs", txt_dir="data/texts"):
        self.pdf_dir = pdf_dir
        self.txt_dir = txt_dir
        self.supported_extensions = {
            'pdf': ['.pdf'],
            'text': ['.txt', '.text', '.md']
        }
```

### 2. 修改文本召回模块 (text_main_batch_inference_enhanced.py)

集成了增强文件处理器：
- 根据输入路径智能选择处理方式
- 支持同时处理PDF和文本文件
- 保持与原有系统的兼容性

```python
if ENHANCED_PROCESSOR_AVAILABLE and ("pdf" in txt_path.lower() or "text" in txt_path.lower()):
    processor = EnhancedFileProcessor()
    pdf_results, txt_results = await processor.process_directory(txt_path)
    retrieval_data = processor.prepare_for_retrieval(pdf_results, txt_results)
```

### 3. 智能路径处理 (run_pipeline.py)

在流水线中添加了智能路径判断：
- 如果PDF目录不存在，自动切换到文本目录
- 如果PDF目录为空但文本目录有文件，自动切换
- 保持灵活性和容错性

```python
if input_path == "data/pdfs" and not os.path.exists(input_path):
    if os.path.exists("data/texts"):
        logger.info("PDF目录不存在，切换到文本目录: data/texts")
        actual_input_path = "data/texts"
```

## 使用方法

### 1. 处理PDF文件
```bash
# 将PDF文件放入 data/pdfs 目录
mkdir -p data/pdfs
cp your_file.pdf data/pdfs/

# 运行流水线
./quick_start.sh --pipeline --domain semiconductor
```

### 2. 处理文本文件
```bash
# 将文本文件放入 data/texts 目录
mkdir -p data/texts
cp your_file.txt data/texts/

# 运行流水线
./quick_start.sh --pipeline --domain semiconductor
```

### 3. 混合处理
系统会自动检测并处理两种类型的文件：
- PDF文件从 `data/pdfs` 加载
- 文本文件从 `data/texts` 加载

## 测试验证

运行测试脚本验证功能：
```bash
# 测试增强处理器
python3 test_enhanced_processor.py

# 测试文本召回
python3 test_text_retrieval.py
```

## 注意事项

1. **PDF库依赖**：
   - 系统支持多种PDF库，但至少需要安装一个
   - 推荐安装：`pip install PyPDF2 pymupdf pdfplumber`

2. **文件编码**：
   - 文本文件支持UTF-8和GBK编码
   - 自动检测并处理编码问题

3. **性能优化**：
   - 批量处理文件以提高效率
   - 支持断点续传（read_hist参数）

## 扩展功能

未来可以添加：
1. 支持更多文件格式（docx、epub等）
2. OCR功能处理扫描版PDF
3. 多语言支持
4. 文件内容预处理和清洗