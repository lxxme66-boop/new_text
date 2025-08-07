# 半导体显示技术领域智能QA生成系统

## 项目概述

本项目是一个专门针对半导体显示技术领域的智能问答生成系统，整合了文本质量评估、问题生成、答案验证等功能。系统支持从学术论文和技术文档中自动生成高质量的逻辑推理问题。

## 核心功能

### 1. 文本处理与过滤
- **智能文本过滤**：自动过滤参考文献、致谢、目录等无关内容
- **中文内容检测**：确保文本包含足够的中文内容
- **学术论文清洗**：去除论文中的格式化内容和元数据

### 2. 质量评估系统
- **四维度评分标准**：
  - 问题完整性（0-2分）
  - 问题复杂性和技术深度（0-2分）
  - 技术正确性和准确性（-1-1分）
  - 思维和推理能力（-1-2分）
- **自动质量判定**：基于评分自动判断文本是否适合生成推理问题

### 3. 问题生成
- **专业问题模板**：针对半导体显示技术领域的专业问题生成
- **逻辑推理导向**：生成需要深度推理才能解答的问题
- **背景独立性**：确保问题不依赖原文即可理解

### 4. 模型支持
- **API模型**：支持OpenAI兼容API（ARK）
- **本地模型**：
  - Ollama支持
  - vLLM高性能推理（支持QwQ-32B等大模型）
  - 批量推理优化

## 项目结构

```
/workspace/
├── TextGeneration/          # 文本生成模块
│   ├── text_filter.py      # 文本过滤功能
│   ├── prompts_conf.py     # 提示词配置
│   └── Datageneration.py   # 数据生成主逻辑
├── TextQA/                  # 问答生成模块
│   ├── quality_assessment_templates.py  # 质量评估模板
│   ├── enhanced_quality_scorer.py       # 增强质量评分器
│   └── enhanced_quality_checker.py      # 增强质量检查器
├── LocalModels/             # 本地模型支持
│   ├── ollama_client.py    # Ollama客户端
│   ├── vllm_client.py      # vLLM客户端
│   └── local_model_manager.py  # 模型管理器
├── data/                    # 数据目录
│   ├── texts/              # 输入文本文件
│   ├── pdfs/               # 输入PDF文件
│   └── output/             # 输出结果
├── config.json             # 配置文件
├── requirements.txt        # 依赖包
└── run_scripts/           # 运行脚本
```

## 文件功能说明

### 核心处理文件

1. **text_main_batch_inference_enhanced.py**
   - 主要功能：批量处理文本文件，支持增强文件处理器
   - 输入：文本文件目录
   - 输出：处理后的文本数据

2. **enhanced_file_processor.py**
   - 主要功能：增强型文件处理，支持文本过滤
   - 特性：自动过滤无关内容，支持PDF和文本

3. **text_qa_generation_enhanced.py**
   - 主要功能：生成问答对，支持增强质量检查
   - 特性：集成详细评分标准，自动质量控制

### 质量评估文件

1. **TextQA/quality_assessment_templates.py**
   - 包含所有评估模板和提示词
   - 支持评分模板、问题生成模板、质量检查模板

2. **TextQA/enhanced_quality_scorer.py**
   - 实现四维度质量评分
   - 自动判断文本适合性

### 模型支持文件

1. **LocalModels/vllm_client.py**
   - vLLM高性能推理支持
   - 批量生成优化
   - 支持大模型（如QwQ-32B）

2. **LocalModels/local_model_manager.py**
   - 统一管理不同模型后端
   - 支持动态切换模型

## 安装与配置

### 1. 环境要求
- Python 3.8+
- CUDA 11.8+（如使用GPU）
- 足够的GPU内存（建议32GB+用于大模型）

### 2. 安装依赖

```bash
# 基础依赖
pip install -r requirements.txt

# vLLM支持（可选）
pip install vllm

# Ollama支持（可选）
# 请访问 https://ollama.ai 安装Ollama
```

### 3. 配置文件

编辑 `config.json` 配置系统参数：

```json
{
  "api": {
    "use_local_models": false,  // 是否使用本地模型
    "default_backend": "ark"     // 默认后端：ark/ollama/vllm
  },
  "models": {
    "local_models": {
      "vllm": {
        "model_path": "/path/to/model",  // 模型路径
        "tensor_parallel_size": 4        // GPU并行数
      }
    }
  }
}
```

## 运行流程

### 完整流程（7步）

1. **数据准备**
   - 将文本文件放入 `data/texts/` 目录
   - 将PDF文件放入 `data/pdfs/` 目录

2. **文本预处理与过滤**
   ```bash
   python text_processor.py --input data/texts --output data/output
   ```

3. **文本召回与批量推理**
   ```bash
   python text_main_batch_inference_enhanced.py \
     --txt_path data/texts \
     --storage_folder data/output \
     --parallel_batch_size 100
   ```

4. **数据清洗**
   ```bash
   python clean_text_data.py \
     --input_file data/output/total_response.pkl \
     --output_file data/output
   ```

5. **QA生成**
   ```bash
   python text_qa_generation_enhanced.py \
     --file_path data/output/total_response.json \
     --output_file data/qa_results \
     --enhanced_quality true
   ```

6. **质量检查**
   ```bash
   python text_qa_generation_enhanced.py \
     --check_task true \
     --file_path data/qa_results/results_343.json \
     --quality_threshold 0.7
   ```

7. **最终输出整理**
   ```bash
   python argument_data.py \
     --input data/qa_results \
     --output data/final_output
   ```

### 一键运行脚本

```bash
# 运行完整流程
bash run_full_pipeline.sh

# 仅处理文本
bash run_text_only.sh
```

## 使用vLLM加速

如果需要使用vLLM进行高性能推理：

1. **启动vLLM服务**（可选）
   ```bash
   python -m vllm.entrypoints.openai.api_server \
     --model /path/to/QwQ-32B \
     --tensor-parallel-size 4 \
     --gpu-memory-utilization 0.95
   ```

2. **配置使用vLLM**
   ```json
   {
     "api": {
       "use_local_models": true,
       "default_backend": "vllm"
     }
   }
   ```

## 输出格式

系统生成的问答对格式：

```json
{
  "source_file": "paper_001.txt",
  "qa_pairs": [
    {
      "question": "在IGZO-TFT器件中，当氧空位浓度从1×10^17 cm^-3增加到1×10^18 cm^-3时，阈值电压会如何变化？这种变化的物理机制是什么？",
      "answer": "阈值电压会负向偏移...",
      "question_type": "reasoning",
      "quality_score": 0.85
    }
  ],
  "metadata": {
    "domain": "semiconductor",
    "processed_at": "2024-01-01T12:00:00"
  }
}
```

## 注意事项

1. **GPU内存管理**
   - 使用vLLM时注意设置合适的 `gpu_memory_utilization`
   - 批处理大小根据GPU内存调整

2. **文本质量**
   - 系统会自动过滤低质量文本
   - 确保输入文本为学术论文或技术文档

3. **模型选择**
   - 大模型（如QwQ-32B）效果更好但需要更多资源
   - 可以根据需求选择不同规模的模型

## 故障排除

1. **内存不足**
   - 减小批处理大小
   - 降低最大模型长度
   - 使用更小的模型

2. **模型加载失败**
   - 检查模型路径是否正确
   - 确认模型文件完整
   - 验证CUDA版本兼容性

3. **质量评分过低**
   - 检查输入文本质量
   - 调整质量阈值
   - 使用更好的模型

## 贡献指南

欢迎提交Issue和Pull Request来改进系统。

## 许可证

本项目采用MIT许可证。