#!/bin/bash
# 半导体显示技术领域智能QA生成系统 - 分步运行脚本

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[信息]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[成功]${NC} $1"
}

print_error() {
    echo -e "${RED}[错误]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[警告]${NC} $1"
}

# 显示菜单
show_menu() {
    echo ""
    echo "=========================================="
    echo "半导体显示技术领域智能QA生成系统"
    echo "分步运行菜单"
    echo "=========================================="
    echo "0. 环境准备与检查"
    echo "1. 文本预处理与过滤"
    echo "2. 文本召回与批量推理"
    echo "3. 数据清洗"
    echo "4. QA生成"
    echo "5. 质量检查"
    echo "6. 数据增强（可选）"
    echo "7. 最终输出整理"
    echo "8. 查看统计报告"
    echo "9. 运行完整流程"
    echo "q. 退出"
    echo "=========================================="
}

# 环境准备
step_0() {
    print_info "执行环境准备与检查..."
    
    # 设置环境变量
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    
    # 创建必要的目录
    mkdir -p data/texts data/pdfs data/output data/qa_results data/final_output logs
    print_success "目录创建完成"
    
    # 检查Python环境
    if command -v python &> /dev/null; then
        print_success "Python环境正常: $(python --version)"
    else
        print_error "未找到Python环境"
        return 1
    fi
    
    # 检查必要的文件
    required_files=(
        "text_processor.py"
        "text_main_batch_inference_enhanced.py"
        "clean_text_data.py"
        "text_qa_generation_enhanced.py"
        "config.json"
    )
    
    missing_files=()
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            missing_files+=("$file")
        fi
    done
    
    if [ ${#missing_files[@]} -eq 0 ]; then
        print_success "所有必要文件都存在"
    else
        print_error "缺少以下文件:"
        for file in "${missing_files[@]}"; do
            echo "  - $file"
        done
        return 1
    fi
    
    # 检查输入文件
    txt_count=$(find data/texts -name "*.txt" 2>/dev/null | wc -l)
    pdf_count=$(find data/pdfs -name "*.pdf" 2>/dev/null | wc -l)
    
    if [ $txt_count -eq 0 ] && [ $pdf_count -eq 0 ]; then
        print_warning "未找到输入文件"
        print_info "请将文本文件放入 data/texts/ 目录"
        print_info "或将PDF文件放入 data/pdfs/ 目录"
    else
        print_success "找到 $txt_count 个文本文件, $pdf_count 个PDF文件"
    fi
    
    # 检查GPU（可选）
    if command -v nvidia-smi &> /dev/null; then
        print_info "GPU信息:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        print_warning "未检测到GPU，将使用CPU运行"
    fi
}

# 步骤1: 文本预处理
step_1() {
    print_info "执行文本预处理与过滤..."
    python text_processor.py \
        --input data/texts \
        --output data/output \
        --batch-size 100 \
        --index 9
    
    if [ $? -eq 0 ]; then
        print_success "文本预处理完成"
    else
        print_error "文本预处理失败"
        return 1
    fi
}

# 步骤2: 文本召回
step_2() {
    print_info "执行文本召回与批量推理..."
    python text_main_batch_inference_enhanced.py \
        --txt_path data/texts \
        --storage_folder data/output \
        --parallel_batch_size 100 \
        --selected_task_number 1000 \
        --index 43
    
    if [ $? -eq 0 ]; then
        print_success "文本召回完成"
    else
        print_error "文本召回失败"
        return 1
    fi
}

# 步骤3: 数据清洗
step_3() {
    print_info "执行数据清洗..."
    
    if [ ! -f "data/output/total_response.pkl" ]; then
        print_error "未找到输入文件: data/output/total_response.pkl"
        print_info "请先执行步骤2"
        return 1
    fi
    
    python clean_text_data.py \
        --input_file data/output/total_response.pkl \
        --output_file data/output
    
    if [ $? -eq 0 ]; then
        print_success "数据清洗完成"
    else
        print_error "数据清洗失败"
        return 1
    fi
}

# 步骤4: QA生成
step_4() {
    print_info "执行QA生成..."
    
    if [ ! -f "data/output/total_response.json" ]; then
        print_error "未找到输入文件: data/output/total_response.json"
        print_info "请先执行步骤3"
        return 1
    fi
    
    python text_qa_generation_enhanced.py \
        --file_path data/output/total_response.json \
        --output_file data/qa_results \
        --index 343 \
        --pool_size 100 \
        --enhanced_quality true
    
    if [ $? -eq 0 ]; then
        print_success "QA生成完成"
    else
        print_error "QA生成失败"
        return 1
    fi
}

# 步骤5: 质量检查
step_5() {
    print_info "执行质量检查..."
    
    if [ ! -f "data/qa_results/results_343.json" ]; then
        print_error "未找到输入文件: data/qa_results/results_343.json"
        print_info "请先执行步骤4"
        return 1
    fi
    
    python text_qa_generation_enhanced.py \
        --check_task true \
        --file_path data/qa_results/results_343.json \
        --output_file data/qa_results \
        --quality_threshold 0.7 \
        --check_times 9
    
    if [ $? -eq 0 ]; then
        print_success "质量检查完成"
    else
        print_error "质量检查失败"
        return 1
    fi
}

# 步骤6: 数据增强
step_6() {
    print_info "执行数据增强..."
    
    if [ ! -f "argument_data.py" ]; then
        print_warning "未找到数据增强脚本，跳过此步骤"
        return 0
    fi
    
    python argument_data.py \
        --input_file data/qa_results/results_343.json \
        --output_file data/qa_results/enhanced_results.json \
        --index 45
    
    if [ $? -eq 0 ]; then
        print_success "数据增强完成"
    else
        print_warning "数据增强失败，但不影响主流程"
    fi
}

# 步骤7: 最终输出整理
step_7() {
    print_info "执行最终输出整理..."
    
    # 确保输出目录存在
    mkdir -p data/final_output
    
    # 复制最终结果
    if [ -f "data/qa_results/enhanced_results.json" ]; then
        cp data/qa_results/enhanced_results.json data/final_output/final_qa_pairs.json
        print_info "使用增强后的结果"
    elif [ -f "data/qa_results/results_343.json" ]; then
        cp data/qa_results/results_343.json data/final_output/final_qa_pairs.json
        print_info "使用原始QA结果"
    else
        print_error "未找到QA结果文件"
        return 1
    fi
    
    # 生成统计报告
    python generate_statistics.py
    
    print_success "最终输出整理完成"
    print_info "结果保存在: data/final_output/"
}

# 查看统计报告
step_8() {
    print_info "查看统计报告..."
    
    if [ ! -f "data/final_output/statistics_report.json" ]; then
        print_error "未找到统计报告"
        print_info "请先执行步骤7"
        return 1
    fi
    
    python -c "
import json

with open('data/final_output/statistics_report.json', 'r', encoding='utf-8') as f:
    report = json.load(f)

print('\\n===== 统计报告 =====')
print(f'总QA对数: {report.get(\"total_qa_pairs\", 0)}')
print(f'源文件数: {report.get(\"total_source_files\", 0)}')
print(f'平均每文件QA数: {report.get(\"average_qa_per_file\", 0):.1f}')

if 'question_type_distribution' in report:
    print('\\n问题类型分布:')
    for q_type, count in report['question_type_distribution'].items():
        print(f'  {q_type}: {count}')
"
}

# 运行完整流程
step_9() {
    print_info "运行完整流程..."
    
    # 依次执行所有步骤
    for i in {0..7}; do
        echo ""
        print_info "执行步骤 $i..."
        step_$i
        if [ $? -ne 0 ]; then
            print_error "步骤 $i 执行失败，流程中断"
            return 1
        fi
    done
    
    print_success "完整流程执行完成！"
}

# 创建统计脚本
cat > generate_statistics.py << 'EOF'
import json
import os

# 读取最终结果
result_file = 'data/final_output/final_qa_pairs.json'
if not os.path.exists(result_file):
    print("错误: 未找到结果文件")
    exit(1)

with open(result_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 统计信息
total_qa_pairs = 0
source_files = set()
question_types = {}
quality_scores = []

for item in data:
    if 'qa_pairs' in item:
        total_qa_pairs += len(item['qa_pairs'])
        source_files.add(item.get('source_file', 'unknown'))
        
        for qa in item['qa_pairs']:
            q_type = qa.get('question_type', 'unknown')
            question_types[q_type] = question_types.get(q_type, 0) + 1
            
            if 'quality_score' in qa:
                quality_scores.append(qa['quality_score'])

# 计算平均质量分数
avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

# 生成报告
report = {
    'total_qa_pairs': total_qa_pairs,
    'total_source_files': len(source_files),
    'question_type_distribution': question_types,
    'average_qa_per_file': total_qa_pairs / len(source_files) if source_files else 0,
    'average_quality_score': avg_quality,
    'quality_score_count': len(quality_scores)
}

# 保存报告
with open('data/final_output/statistics_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f'统计报告已生成: data/final_output/statistics_report.json')
EOF

# 主循环
while true; do
    show_menu
    read -p "请选择操作 (0-9, q): " choice
    
    case $choice in
        0) step_0 ;;
        1) step_1 ;;
        2) step_2 ;;
        3) step_3 ;;
        4) step_4 ;;
        5) step_5 ;;
        6) step_6 ;;
        7) step_7 ;;
        8) step_8 ;;
        9) step_9 ;;
        q|Q) 
            print_info "退出程序"
            exit 0 
            ;;
        *)
            print_error "无效的选择，请重新输入"
            ;;
    esac
    
    echo ""
    read -p "按回车键继续..."
done