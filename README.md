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

### 通用流程（7步骤）vs 半导体专用流程（3步骤）

系统提供两种处理流程：

#### 1. 通用QA生成流程（7步骤）

完整的通用流程包含以下7个步骤：

1. **文本预处理与过滤** (`text_processor.py`)
2. **文本召回与批量推理** (`text_main_batch_inference_enhanced.py`)
3. **数据清洗** (`clean_text_data.py`)
4. **QA生成** (`text_qa_generation_enhanced.py`)
5. **质量检查** (`text_qa_generation_enhanced.py --check_task`)
6. **数据增强** (`argument_data.py`)
7. **最终输出整理**

#### 2. 半导体专用流程（3步骤）

针对半导体领域优化的精简流程：

1. **文本预处理** (`text_processor.py`)
2. **核心QA生成** (`semiconductor_qa_generator.py`)
3. **数据增强与重写** (`argument_data.py`)

#### 流程对比

| 步骤 | 通用流程 | 半导体专用流程 | 说明 |
|------|----------|----------------|------|
| 1 | 文本预处理与过滤 | 文本预处理 | 半导体流程集成了过滤功能 |
| 2 | 文本召回与批量推理 | - | 半导体流程内置在QA生成中 |
| 3 | 数据清洗 | - | 半导体流程在预处理中完成 |
| 4 | QA生成 | 核心QA生成 | 半导体流程包含质量评估 |
| 5 | 质量检查 | - | 半导体流程内置质量检查 |
| 6 | 数据增强 | 数据增强与重写 | 功能相同 |
| 7 | 最终输出整理 | - | 半导体流程自动整理输出 |

### 完整处理流程图

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

### 1. 半导体专用流程（推荐）

#### 一键运行

```bash
# 使用默认参数运行（精简3步骤模式）
python run_semiconductor_qa.py --input-dir data/texts --output-dir data/qa_results

# 启用完整7步骤模式
python run_semiconductor_qa.py --input-dir data/texts --output-dir data/qa_results --enable-full-steps

# 自定义参数运行
python run_semiconductor_qa.py \
    --input-dir /path/to/texts \
    --output-dir /path/to/output \
    --model qwq_32 \
    --batch-size 32 \
    --gpu-devices "0,1,2,3" \
    --enable-full-steps  # 可选：启用完整7步骤
```

**注意**：
- 默认运行精简3步骤模式（更快速）
- 使用 `--enable-full-steps` 参数启用完整7步骤模式（更全面）
- 完整模式包含：文本召回、数据清洗、独立质量检查和详细统计报告

#### 分步运行

```bash
# 步骤1: 文本预处理
python text_processor.py --input data/texts --output data/output

# 步骤2: 核心QA生成（包含质量评估）
python -c "
import asyncio
from semiconductor_qa_generator import run_semiconductor_qa_generation

async def main():
    results = await run_semiconductor_qa_generation(
        raw_folders=['data/output/preprocessed_texts.json'],
        save_paths=['data/qa_results/qa_generated.json'],
        model_name='qwq_32',
        batch_size=32,
        gpu_devices='4,5,6,7'
    )
    print(f'生成了 {len(results)} 个QA对')

asyncio.run(main())
"

# 步骤3: 数据增强与重写
python -c "
import asyncio
import json
from argument_data import ArgumentDataProcessor

async def main():
    processor = ArgumentDataProcessor()
    with open('data/qa_results/qa_generated.json', 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    enhanced_data = await processor.enhance_qa_data(qa_data)
    with open('data/qa_results/final_qa_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(enhanced_data, f, ensure_ascii=False, indent=2)
    print(f'增强完成，最终数据集包含 {len(enhanced_data)} 个QA对')

asyncio.run(main())
"
```

### 2. 通用流程（完整7步骤）

#### 一键运行

```bash
# 运行完整流程脚本
bash run_full_pipeline.sh

# 或使用Python统一入口
python run_pipeline.py --input_path data/texts --domain semiconductor
```

#### 分步运行

```bash
# 步骤1: 文本预处理与过滤
python text_processor.py \
    --input data/texts \
    --output data/output \
    --batch-size 100 \
    --index 9

# 步骤2: 文本召回与批量推理
python text_main_batch_inference_enhanced.py \
    --txt_path data/texts \
    --storage_folder data/output \
    --parallel_batch_size 100 \
    --selected_task_number 1000 \
    --index 43

# 步骤3: 数据清洗
python clean_text_data.py \
    --input_file data/output/total_response.pkl \
    --output_file data/output

# 步骤4: QA生成
python text_qa_generation_enhanced.py \
    --file_path data/output/total_response.json \
    --output_file data/qa_results \
    --index 343 \
    --pool_size 100 \
    --enhanced_quality true

# 步骤5: 质量检查
python text_qa_generation_enhanced.py \
    --check_task true \
    --file_path data/qa_results/results_343.json \
    --output_file data/qa_results \
    --quality_threshold 0.7 \
    --check_times 9

# 步骤6: 数据增强（可选）
python argument_data.py \
    --input_file data/qa_results/results_343.json \
    --output_file data/qa_results/enhanced_results.json \
    --index 45

# 步骤7: 最终输出整理
# 复制并生成统计报告
cp data/qa_results/results_343.json data/final_output/final_qa_pairs.json
python generate_statistics.py --input data/final_output/final_qa_pairs.json
```

### 3. 选择建议

- **半导体专用流程**：针对半导体领域优化，集成度高，运行效率更高，推荐用于半导体相关文档
- **通用流程**：适用于各种领域，步骤可控性强，适合需要精细控制每个步骤的场景

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