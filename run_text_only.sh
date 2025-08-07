#!/bin/bash
# 半导体显示技术领域智能QA生成系统 - 仅文本处理脚本

echo "=========================================="
echo "文本专用快速处理流程"
echo "=========================================="

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 创建必要的目录
mkdir -p data/texts data/output data/qa_results logs

# 检查文本文件
if [ ! "$(ls -A data/texts/*.txt 2>/dev/null)" ]; then
    echo "错误: 请将文本文件放入 data/texts/ 目录"
    exit 1
fi

echo "找到文本文件: $(ls -1 data/texts/*.txt | wc -l) 个"

# 步骤1: 文本召回与批量推理（跳过预处理，直接使用增强处理器）
echo ""
echo "[步骤1] 文本召回与批量推理..."
python text_main_batch_inference_enhanced.py \
    --txt_path data/texts \
    --storage_folder data/output \
    --parallel_batch_size 100 \
    --index 43

# 步骤2: 数据清洗
echo ""
echo "[步骤2] 数据清洗..."
python clean_text_data.py \
    --input_file data/output/total_response.pkl \
    --output_file data/output

# 步骤3: 快速QA生成（使用默认质量设置）
echo ""
echo "[步骤3] QA生成..."
python text_qa_generation_enhanced.py \
    --file_path data/output/total_response.json \
    --output_file data/qa_results \
    --index 343 \
    --pool_size 100

echo ""
echo "=========================================="
echo "文本处理完成！"
echo "结果保存在: data/qa_results/results_343.json"
echo "=========================================="