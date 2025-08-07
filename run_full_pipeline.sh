#!/bin/bash
# 半导体显示技术领域智能QA生成系统 - 完整流程运行脚本

echo "=========================================="
echo "半导体显示技术领域智能QA生成系统"
echo "完整流程运行脚本"
echo "=========================================="

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES="0,1,2,3"  # 根据实际情况调整GPU

# 创建必要的目录
echo "[步骤0] 创建必要的目录..."
mkdir -p data/texts data/pdfs data/output data/qa_results data/final_output logs

# 检查是否有输入文件
if [ ! "$(ls -A data/texts/*.txt 2>/dev/null)" ] && [ ! "$(ls -A data/pdfs/*.pdf 2>/dev/null)" ]; then
    echo "错误: 请将文本文件放入 data/texts/ 目录或PDF文件放入 data/pdfs/ 目录"
    exit 1
fi

# 步骤1: 文本预处理与过滤
echo ""
echo "[步骤1] 文本预处理与过滤..."
python text_processor.py \
    --input data/texts \
    --output data/output \
    --batch-size 100 \
    --index 9

if [ $? -ne 0 ]; then
    echo "错误: 文本预处理失败"
    exit 1
fi

# 步骤2: 文本召回与批量推理
echo ""
echo "[步骤2] 文本召回与批量推理..."
python text_main_batch_inference_enhanced.py \
    --txt_path data/texts \
    --storage_folder data/output \
    --parallel_batch_size 100 \
    --selected_task_number 1000 \
    --index 43

if [ $? -ne 0 ]; then
    echo "错误: 文本召回失败"
    exit 1
fi

# 步骤3: 数据清洗
echo ""
echo "[步骤3] 数据清洗..."
python clean_text_data.py \
    --input_file data/output/total_response.pkl \
    --output_file data/output

if [ $? -ne 0 ]; then
    echo "错误: 数据清洗失败"
    exit 1
fi

# 步骤4: QA生成
echo ""
echo "[步骤4] QA生成..."
python text_qa_generation_enhanced.py \
    --file_path data/output/total_response.json \
    --output_file data/qa_results \
    --index 343 \
    --pool_size 100 \
    --enhanced_quality true

if [ $? -ne 0 ]; then
    echo "错误: QA生成失败"
    exit 1
fi

# 步骤5: 质量检查
echo ""
echo "[步骤5] 质量检查..."
python text_qa_generation_enhanced.py \
    --check_task true \
    --file_path data/qa_results/results_343.json \
    --output_file data/qa_results \
    --quality_threshold 0.7 \
    --check_times 9

if [ $? -ne 0 ]; then
    echo "错误: 质量检查失败"
    exit 1
fi

# 步骤6: 数据增强（可选）
echo ""
echo "[步骤6] 数据增强..."
if [ -f "argument_data.py" ]; then
    python argument_data.py \
        --input_file data/qa_results/results_343.json \
        --output_file data/qa_results/enhanced_results.json \
        --index 45
else
    echo "跳过数据增强步骤"
fi

# 步骤7: 最终输出整理
echo ""
echo "[步骤7] 最终输出整理..."
# 复制最终结果
cp data/qa_results/results_343.json data/final_output/final_qa_pairs.json

# 生成统计报告
python -c "
import json
import os

# 读取最终结果
with open('data/final_output/final_qa_pairs.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 统计信息
total_qa_pairs = 0
source_files = set()
question_types = {}

for item in data:
    if 'qa_pairs' in item:
        total_qa_pairs += len(item['qa_pairs'])
        source_files.add(item.get('source_file', 'unknown'))
        
        for qa in item['qa_pairs']:
            q_type = qa.get('question_type', 'unknown')
            question_types[q_type] = question_types.get(q_type, 0) + 1

# 生成报告
report = {
    'total_qa_pairs': total_qa_pairs,
    'total_source_files': len(source_files),
    'question_type_distribution': question_types,
    'average_qa_per_file': total_qa_pairs / len(source_files) if source_files else 0
}

# 保存报告
with open('data/final_output/statistics_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f'\\n统计报告:')
print(f'总QA对数: {total_qa_pairs}')
print(f'源文件数: {len(source_files)}')
print(f'平均每文件QA数: {report[\"average_qa_per_file\"]:.1f}')
print(f'\\n问题类型分布:')
for q_type, count in question_types.items():
    print(f'  {q_type}: {count}')
"

echo ""
echo "=========================================="
echo "完整流程运行完成！"
echo "最终结果保存在: data/final_output/"
echo "=========================================="