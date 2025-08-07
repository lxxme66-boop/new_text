# 文本与PDF分离处理解决方案

## 问题分析

### 原始问题
在 `new_text` 项目中，txt文件放在生成的 `pdf` 目录下可以加载，但放在 `text` 目录下无法加载。

### 问题原因
1. **变量命名混淆**：代码中使用了 `pdf_path` 变量来处理txt文件，这导致了路径配置的混乱
2. **路径硬编码**：原始代码中的路径配置不够灵活，没有正确区分文本和PDF文件的处理路径
3. **处理逻辑混合**：文本处理和PDF处理逻辑没有明确分离，导致文件类型和路径管理混乱

## 解决方案

### 1. 创建独立的处理器

#### 文本处理器 (`text_processor.py`)
- 专门处理txt文件
- 从 `data/texts` 目录读取文件
- 结果保存到 `data/qa_results/texts`
- 使用 `TextGeneration.Datageneration` 模块进行文本处理

#### PDF处理器 (`pdf_processor_main.py`)
- 专门处理PDF文件
- 从 `data/pdfs` 目录读取文件
- 结果保存到 `data/qa_results/pdfs`
- 使用 `MultiModal.pdf_processor` 模块进行多模态处理

### 2. 更新配置文件

在 `config.json` 中添加了明确的路径配置：

```json
"paths": {
  "text_dir": "data/texts",
  "pdf_dir": "data/pdfs",
  "text_output_dir": "data/qa_results/texts",
  "pdf_output_dir": "data/qa_results/pdfs"
}
```

### 3. 文件组织结构

```
new_text/
├── data/
│   ├── texts/          # 存放txt文件
│   │   ├── sample_optics.txt
│   │   ├── sample_semiconductor.txt
│   │   └── test.txt
│   ├── pdfs/           # 存放PDF文件
│   └── qa_results/
│       ├── texts/      # txt处理结果
│       └── pdfs/       # PDF处理结果
```

## 使用方法

### 处理文本文件

```bash
# 处理所有txt文件
python text_processor.py

# 处理特定txt文件
python text_processor.py --input data/texts/sample.txt

# 处理文件夹中的txt文件
python text_processor.py --input data/texts --batch-size 50
```

### 处理PDF文件

```bash
# 处理所有PDF文件
python pdf_processor_main.py

# 处理特定PDF文件
python pdf_processor_main.py --input data/pdfs/document.pdf

# 启用多模态处理
python pdf_processor_main.py --enable-multimodal
```

### 测试分离处理

```bash
# 运行测试脚本
python test_separated_processing.py
```

## 优势

1. **清晰的职责分离**：文本和PDF处理完全独立，各自有专门的处理器
2. **灵活的配置**：通过配置文件可以轻松修改输入输出路径
3. **易于维护**：代码结构清晰，便于后续扩展和维护
4. **避免混淆**：文件类型和处理逻辑明确分离，不会再出现路径混乱的问题
5. **支持批处理**：两种处理器都支持批量处理和并发处理

## 注意事项

1. 确保txt文件放在 `data/texts` 目录下
2. 确保PDF文件放在 `data/pdfs` 目录下
3. 处理前检查相应的模块是否正确导入
4. 如果使用API处理，确保API配置正确；否则可以使用mock模式进行测试