#!/bin/bash

# 智能文本QA生成系统 - 一键运行脚本
# 支持完整流水线和分步执行

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}[$(date +'%Y-%m-%d %H:%M:%S')] ${message}${NC}"
}

# 打印标题
print_header() {
    echo -e "${BLUE}===============================================${NC}"
    echo -e "${BLUE}     智能文本QA生成系统 v3.0 - 纯文本版${NC}"
    echo -e "${BLUE}===============================================${NC}"
}

# 检查Python环境
check_python() {
    print_message $YELLOW "检查Python环境..."
    
    if ! command -v python3 &> /dev/null; then
        print_message $RED "错误: 未找到Python3"
        exit 1
    fi
    
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    print_message $GREEN "Python版本: $python_version"
}

# 检查GPU
check_gpu() {
    print_message $YELLOW "检查GPU环境..."
    
    if ! command -v nvidia-smi &> /dev/null; then
        print_message $RED "警告: 未找到NVIDIA GPU"
        read -p "是否继续？(y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    fi
}

# 创建必要的目录
setup_directories() {
    print_message $YELLOW "创建目录结构..."
    
    directories=(
        "data/texts"
        "data/output"
        "data/output/retrieved"
        "data/output/cleaned"
        "data/output/qa_results"
        "data/output/quality_checked"
        "data/output/final"
        "logs"
        "temp"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
    done
    
    print_message $GREEN "目录创建完成"
}

# 检查依赖
check_dependencies() {
    print_message $YELLOW "检查依赖包..."
    
    # 检查vLLM
    if python3 -c "import vllm" 2>/dev/null; then
        print_message $GREEN "vLLM已安装"
    else
        print_message $YELLOW "vLLM未安装，将使用模拟模式"
    fi
    
    # 检查其他关键依赖
    dependencies=("asyncio" "json" "torch")
    for dep in "${dependencies[@]}"; do
        if python3 -c "import $dep" 2>/dev/null; then
            print_message $GREEN "$dep 已安装"
        else
            print_message $RED "错误: $dep 未安装"
            print_message $YELLOW "请运行: pip install -r requirements.txt"
            exit 1
        fi
    done
}

# 运行完整流水线
run_full_pipeline() {
    print_message $BLUE "运行完整流水线..."
    
    # 检查输入文件
    if [ ! -d "data/texts" ] || [ -z "$(ls -A data/texts 2>/dev/null)" ]; then
        print_message $YELLOW "data/texts目录为空，创建示例文件..."
        echo "IGZO是一种重要的氧化物半导体材料，具有高迁移率、良好的稳定性和透明性。" > data/texts/sample.txt
        echo "TFT器件的阈值电压稳定性对显示质量有重要影响。" >> data/texts/sample.txt
    fi
    
    # 运行流水线
    python3 run_pipeline_text_only.py \
        --config config_local.json \
        --input data/texts \
        --all
}

# 运行单个步骤
run_single_step() {
    local step=$1
    
    case $step in
        "qa")
            print_message $BLUE "运行QA生成..."
            python3 text_qa_generation_categorized.py \
                --file_path data/texts \
                --output_file data/output \
                --pool_size 10 \
                --config config_local.json
            ;;
        "clean")
            print_message $BLUE "运行数据清理..."
            python3 clean_data.py \
                --input_file data/output/retrieved_*.pkl \
                --output_file data/output/cleaned/cleaned.json
            ;;
        "quality")
            print_message $BLUE "运行质量控制..."
            python3 run_pipeline_text_only.py \
                --input data/output/qa_results \
                --stages quality_control
            ;;
        *)
            print_message $RED "未知步骤: $step"
            exit 1
            ;;
    esac
}

# 显示使用说明
show_usage() {
    echo "使用方法："
    echo "  $0 [选项]"
    echo ""
    echo "选项："
    echo "  all       - 运行完整流水线（默认）"
    echo "  qa        - 只运行QA生成"
    echo "  clean     - 只运行数据清理"
    echo "  quality   - 只运行质量控制"
    echo "  check     - 只检查环境"
    echo "  help      - 显示此帮助信息"
}

# 主函数
main() {
    print_header
    
    # 解析参数
    action=${1:-all}
    
    case $action in
        "help")
            show_usage
            exit 0
            ;;
        "check")
            check_python
            check_gpu
            check_dependencies
            print_message $GREEN "环境检查完成"
            exit 0
            ;;
        "all")
            check_python
            check_gpu
            setup_directories
            check_dependencies
            run_full_pipeline
            ;;
        "qa"|"clean"|"quality")
            check_python
            setup_directories
            check_dependencies
            run_single_step $action
            ;;
        *)
            print_message $RED "未知选项: $action"
            show_usage
            exit 1
            ;;
    esac
    
    print_message $GREEN "执行完成！"
    print_message $YELLOW "结果保存在 data/output/ 目录"
}

# 运行主函数
main "$@"