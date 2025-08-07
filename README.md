# 半导体显示技术QA生成系统

## 项目概述

本项目是一个专门针对半导体显示技术领域的智能问答生成系统，通过深度学习模型自动从技术文献中提取知识点并生成高质量的问答对。

## 主要功能

1. **文本质量评估**：评估输入文档是否适合生成逻辑推理问题
2. **问题生成**：基于文档内容生成需要逻辑推理的技术问题
3. **质量检测**：对生成的问题进行质量评估
4. **数据增强**：对生成的QA数据进行润色和优化

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                         输入文档                              │
└─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    text_processor.py                         │
│                  (文本预处理入口)                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              semiconductor_qa_generator.py                   │
│                  (核心QA生成引擎)                            │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐       │
│  │文本质量评估 │→ │ 问题生成    │→ │ 问题质量评估 │       │
│  └─────────────┘  └─────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   argument_data.py                           │
│                  (数据增强与重写)                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      最终输出                                │
│                 高质量QA数据集                               │
└─────────────────────────────────────────────────────────────┘
```

## 处理流程

### 完整处理流程
```
A[run_pipeline.py/运行脚本] --> B[text_processor.py]
B --> C[enhanced_file_processor.py]
C --> D[text_main_batch_inference_enhanced.py]
D --> E[TextGeneration/Datageneration.py]
D --> F[clean_data.py/clean_text_data.py]
F --> G[text_qa_generation_enhanced.py]
G --> H[semiconductor_qa_generator.py]
H --> I[TextQA/enhanced_quality_checker.py]
I --> J[TextQA/enhanced_quality_scorer.py]
G --> K[argument_data.py]
K --> L[最终输出]

M[LocalModels/local_model_manager.py] --> H
N[LocalModels/vllm_client.py] --> M
O[LocalModels/ollama_client.py] --> M
```

## 环境要求

- Python 3.8+
- CUDA 11.7+ (用于GPU加速)
- 至少32GB内存
- 4个GPU (用于大模型推理)

## 安装指南

1. 克隆项目
```bash
git clone <repository_url>
cd semiconductor-qa-generator
```

2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 安装vLLM (可选，用于本地模型推理)
```bash
pip install vllm
```

## 使用方法

### 1. 推荐方式 - 运行完整的半导体QA生成流程

```bash
python run_semiconductor_qa.py --input-dir data/texts --output-dir data/qa_results
```

这将按照正确的流程依次执行：
1. text_processor.py - 文本预处理
2. semiconductor_qa_generator.py - 核心QA生成
3. argument_data.py - 数据增强与重写

### 2. 旧版方式 - 使用增强版QA生成（当前shell脚本使用的方式）

```bash
# 运行完整流程
bash run_full_pipeline.sh

# 或直接运行Python脚本
python text_qa_generation_enhanced.py --file_path data/output/total_response.json
```

### 3. 分步执行

```bash
# 步骤1: 文本预处理
python text_processor.py --input data/texts --output data/output

# 步骤2: QA生成
python semiconductor_qa_generator.py

# 步骤3: 数据增强
python argument_data.py --input data/qa_results/qa_generated.json
```

### 4. 自定义参数运行

```bash
python run_semiconductor_qa.py \
    --input-dir /path/to/texts \
    --output-dir /path/to/output \
    --model qwq_32 \
    --batch-size 32 \
    --gpu-devices "0,1,2,3"
```

## 配置说明

### 模型配置
支持的模型：
- `qwq_32`: QwQ-32B模型（默认）
- `qw2_72`: Qwen2-72B
- `qw2.5_32`: Qwen2.5-32B
- `qw2.5_72`: Qwen2.5-72B
- `llama3.1_70`: Llama3.1-70B

### 评估标准

#### 文本质量评分标准
1. **问题完整性** (0-2分)
2. **问题复杂性和技术深度** (0-2分)
3. **技术正确性和准确性** (-1-1分)
4. **思维和推理** (-1-2分)

#### 问题质量评估标准
1. **因果性**：展现完整的技术逻辑链
2. **周密性**：科学严谨的思维过程
3. **完整性**：问题独立、自足、语义完整

## 输出格式

生成的QA数据包含以下字段：
```json
{
    "id": "文档ID",
    "paper_content": "原始文档内容",
    "question_li": "生成的问题",
    "reasoning": "推理过程（如有）",
    "answer": "答案（如有）"
}
```

## 项目结构

```
.
├── semiconductor_qa_generator.py  # 核心QA生成器
├── text_processor.py             # 文本处理入口
├── run_pipeline.py              # 完整流程控制
├── argument_data.py             # 数据增强模块
├── TextGeneration/              # 文本生成相关模块
│   ├── prompts_conf.py         # Prompt配置
│   └── Datageneration.py       # 数据生成逻辑
├── TextQA/                      # QA质量评估模块
│   ├── enhanced_quality_checker.py
│   └── quality_assessment_templates.py
├── LocalModels/                 # 本地模型接口
│   ├── local_model_manager.py
│   ├── vllm_client.py
│   └── ollama_client.py
└── data/                        # 数据目录
    ├── texts/                   # 输入文本
    └── output/                  # 输出结果
```

## 注意事项

1. **GPU内存**：运行大模型需要充足的GPU内存，建议使用4张V100或A100
2. **文本长度**：系统会自动处理超长文本，但可能会截断
3. **批处理**：适当调整batch_size以平衡速度和内存使用
4. **模型路径**：确保模型文件路径正确配置

## 故障排除

### 1. vLLM未安装错误
```bash
pip install vllm
```

### 2. CUDA内存不足
- 减小batch_size
- 使用更少的GPU
- 选择较小的模型

### 3. 模型加载失败
- 检查模型路径是否正确
- 确保有足够的磁盘空间
- 验证模型文件完整性

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。

## 许可证

本项目采用MIT许可证。