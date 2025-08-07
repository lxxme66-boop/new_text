# 智能文本QA生成系统 v3.0 - 纯文本处理版

## 系统概述

本系统是一个专注于文本处理的智能问答生成系统，能够从文本数据中自动生成高质量的问答对。系统支持四种问题类型，并按照指定比例生成：

- **事实型问题 (15%)**: 获取指标、数值、性能参数等客观信息
- **比较型问题 (15%)**: 比较不同材料、结构或方案的差异
- **推理型问题 (50%)**: 探究机制原理，解释行为或结果的原因
- **开放型问题 (20%)**: 提供优化建议、改进方法或解决方案

## 主要功能

### 1. 文本数据处理
- 支持多种文本格式输入（txt、json、pkl）
- 自动文本清理和预处理
- 智能内容提取和分段

### 2. 分类问答生成
- 四种问题类型自动分类
- 按预设比例生成不同类型问题
- 专业领域定制（半导体、显示技术等）

### 3. 本地大模型支持
- 支持vLLM高性能推理框架
- 针对2卡A100优化配置
- 支持70B+参数大模型

### 4. 质量控制
- 多维度质量评估
- 自动过滤低质量问答对
- 生成详细的质量报告

## 系统架构

```
智能文本QA生成系统/
├── 核心模块/
│   ├── text_qa_generation_categorized.py  # 分类QA生成
│   ├── run_pipeline_text_only.py          # 主流水线
│   └── clean_data.py                      # 数据清理
├── 本地模型支持/
│   ├── LocalModels/vllm_client.py         # vLLM客户端
│   └── LocalModels/ollama_client.py       # Ollama客户端
├── 配置文件/
│   ├── config_local.json                  # 本地模型配置
│   └── requirements.txt                   # 依赖包列表
└── 数据目录/
    ├── data/texts/                        # 输入文本
    └── data/output/                       # 输出结果
```

## 环境要求

### 硬件要求
- GPU: 2张NVIDIA A100 (80GB) 或同等算力GPU
- 内存: 128GB+ RAM
- 存储: 500GB+ SSD

### 软件要求
- Python 3.8+
- CUDA 11.8+
- PyTorch 2.0+
- vLLM 0.2.0+

## 安装步骤

### 1. 克隆项目
```bash
git clone <repository_url>
cd text_qa_generation_system
```

### 2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 安装vLLM（用于本地大模型）
```bash
pip install vllm
```

### 5. 下载模型（可选）
如果使用本地大模型，需要下载模型文件：
```bash
# 示例：下载Qwen-72B模型
huggingface-cli download Qwen/Qwen-72B-Chat --local-dir /path/to/models/Qwen-72B-Chat
```

## 配置说明

### 1. 修改配置文件
编辑 `config_local.json`，设置模型路径和参数：

```json
{
  "models": {
    "local_models": {
      "vllm": {
        "enabled": true,
        "model_path": "/path/to/your/model",
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.9
      }
    }
  }
}
```

### 2. 准备输入数据
将文本文件放入 `data/texts/` 目录：
```bash
mkdir -p data/texts
cp your_documents.txt data/texts/
```

## 运行流程

### 方式一：一键运行（推荐）
```bash
# 运行完整流水线
./run_all.sh

# 或使用Python直接运行
python run_pipeline_text_only.py --input data/texts --all
```

### 方式二：分步运行

#### 1. 文本召回（可选）
```bash
python text_main_batch_inference_enhanced.py \
    --txt_path data/texts \
    --index 43 \
    --pool_size 100
```

#### 2. 数据清理（可选）
```bash
python clean_data.py \
    --input_file data/retrieved/retrieved_*.pkl \
    --output_file data/cleaned/cleaned.json
```

#### 3. QA生成（核心步骤）
```bash
python text_qa_generation_categorized.py \
    --file_path data/texts/your_file.txt \
    --output_file data/output \
    --pool_size 10
```

#### 4. 质量控制（可选）
```bash
python run_pipeline_text_only.py \
    --input data/output/categorized_results_1.json \
    --stages quality_control
```

### 方式三：自定义流水线
```bash
# 只运行QA生成和质量控制
python run_pipeline_text_only.py \
    --input data/texts \
    --stages qa_generation quality_control
```

## 输出说明

### 1. QA结果文件
位置：`data/output/categorized_results_*.json`

格式示例：
```json
[
  {
    "question_type": "factual",
    "question": "IGZO TFT的典型迁移率范围是多少？",
    "answer": "IGZO TFT的典型载流子迁移率在10-50 cm²/V·s范围内...",
    "content": "原始文本片段...",
    "generated": true
  }
]
```

### 2. 统计报告
位置：`data/output/categorized_results_*_stats.json`

包含内容：
- 总生成数量
- 各类型问题数量和比例
- 实际比例vs预期比例对比

### 3. 质量报告
位置：`data/output/quality_checked/quality_report.json`

包含内容：
- 质量评分分布
- 通过率统计
- 问题类型质量分析

## 常见问题

### Q1: 如何调整问题类型比例？
修改 `config_local.json` 中的配置：
```json
"question_type_ratios": {
  "factual": 0.15,
  "comparison": 0.15,
  "reasoning": 0.50,
  "open_ended": 0.20
}
```

### Q2: GPU内存不足怎么办？
1. 减小批处理大小：`--pool_size 5`
2. 降低GPU内存使用率：修改 `gpu_memory_utilization` 为 0.8
3. 使用更小的模型

### Q3: 如何添加新的专业领域？
在 `text_qa_generation_categorized.py` 中修改prompt模板，添加领域特定的示例和指导。

### Q4: 生成速度太慢？
1. 增加批处理大小
2. 使用更多GPU：修改 `tensor_parallel_size`
3. 启用模型量化

## 性能优化建议

1. **批处理优化**
   - 根据GPU内存调整 `pool_size`
   - 建议A100使用 20-50 的批大小

2. **模型优化**
   - 使用Flash Attention加速
   - 启用混合精度训练
   - 考虑模型量化（int8/int4）

3. **数据预处理**
   - 提前清理和格式化文本
   - 控制输入文本长度
   - 使用缓存机制

## 扩展功能

1. **支持更多模型**
   - 修改 `LocalModels/` 添加新的模型客户端
   - 支持Hugging Face模型

2. **自定义问题类型**
   - 在 `QuestionType` 枚举中添加新类型
   - 定义新的prompt模板

3. **多语言支持**
   - 修改prompt模板支持其他语言
   - 调整分词和处理逻辑

## 许可证

本项目采用 Apache 2.0 许可证。

## 联系方式

如有问题或建议，请提交Issue或联系维护团队。

---

**注意**: 本系统专注于文本处理，已移除多模态相关功能。如需处理PDF或图像，请使用其他专门工具。