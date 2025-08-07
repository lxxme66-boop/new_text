#!/bin/bash

# 半导体QA生成流程脚本
# 按照正确的流程调用各个模块

echo "================================"
echo "半导体QA生成系统"
echo "================================"

# 设置默认参数
INPUT_DIR="${1:-data/texts}"
OUTPUT_DIR="${2:-data/qa_results}"
MODEL="${3:-qwq_32}"
BATCH_SIZE="${4:-32}"
GPU_DEVICES="${5:-4,5,6,7}"

# 显示配置
echo ""
echo "配置信息:"
echo "- 输入目录: $INPUT_DIR"
echo "- 输出目录: $OUTPUT_DIR"
echo "- 模型: $MODEL"
echo "- 批处理大小: $BATCH_SIZE"
echo "- GPU设备: $GPU_DEVICES"
echo ""

# 检查输入目录
if [ ! -d "$INPUT_DIR" ]; then
    echo "错误: 输入目录 $INPUT_DIR 不存在"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行主流程
echo "开始运行QA生成流程..."
python run_semiconductor_qa.py \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --model "$MODEL" \
    --batch-size "$BATCH_SIZE" \
    --gpu-devices "$GPU_DEVICES"

# 检查运行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "================================"
    echo "QA生成流程完成!"
    echo "结果保存在: $OUTPUT_DIR"
    echo "================================"
    
    # 显示统计信息
    if [ -f "$OUTPUT_DIR/pipeline_stats.json" ]; then
        echo ""
        echo "统计信息:"
        cat "$OUTPUT_DIR/pipeline_stats.json"
    fi
else
    echo ""
    echo "错误: QA生成流程失败"
    exit 1
fi