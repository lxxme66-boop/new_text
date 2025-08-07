#!/bin/bash

# 智能文本QA生成系统 - 整合版快速启动脚本
# 版本: 3.0.0
# 功能: 完整数据流水线演示，包括文本召回、QA生成、质量控制、数据改写增强

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}[$(date +'%Y-%m-%d %H:%M:%S')] ${message}${NC}"
}

print_header() {
    echo -e "${CYAN}"
    echo "==============================================="
    echo "  智能文本QA生成系统 - 整合版"
    echo "  版本: 3.0.0"
    echo "  功能完整的智能问答生成流水线"
    echo "==============================================="
    echo -e "${NC}"
}

print_usage() {
    echo -e "${WHITE}使用方法:${NC}"
    echo "  $0 [选项]"
    echo ""
    echo -e "${WHITE}选项:${NC}"
    echo "  --demo                    运行完整演示流水线"
    echo "  --install                 安装依赖包"
    echo "  --check                   检查环境和配置"
    echo "  --local                   启动本地模型服务"
    echo "  --pipeline               运行完整流水线"
    echo "  --text-retrieval         仅运行文本召回"
    echo "  --qa-generation          仅运行QA生成"
    echo "  --quality-control        仅运行质量控制"
    echo "  --data-rewriting         仅运行数据改写"
    echo "  --multimodal             启用多模态处理"
    echo "  --domain [DOMAIN]        指定专业领域 (semiconductor/optics/materials)"
    echo "  --help, -h               显示此帮助信息"
    echo ""
    echo -e "${WHITE}示例:${NC}"
    echo "  $0 --demo                # 运行完整演示"
    echo "  $0 --pipeline --domain semiconductor --multimodal"
    echo "  $0 --qa-generation --domain optics"
}

# 检查Python环境
check_python() {
    print_message $BLUE "检查Python环境..."
    
    if ! command -v python3 &> /dev/null; then
        print_message $RED "错误: 未找到Python3，请先安装Python 3.8+"
        exit 1
    fi
    
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_message $GREEN "Python版本: $python_version"
    
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        print_message $RED "错误: Python版本过低，需要Python 3.8+"
        exit 1
    fi
}

# 安装依赖
install_dependencies() {
    print_message $BLUE "安装依赖包..."
    
    if [ ! -f "requirements.txt" ]; then
        print_message $RED "错误: 未找到requirements.txt文件"
        exit 1
    fi
    
    print_message $YELLOW "安装基础依赖..."
    pip3 install -r requirements.txt
    
    print_message $YELLOW "安装多模态支持..."
    pip3 install PyMuPDF Pillow opencv-python
    
    print_message $YELLOW "安装本地模型支持（可选）..."
    pip3 install ollama transformers torch torchvision || print_message $YELLOW "警告: 本地模型依赖安装失败，将跳过"
    
    print_message $GREEN "依赖安装完成！"
}

# 检查配置文件
check_config() {
    print_message $BLUE "检查配置文件..."
    
    if [ ! -f "config.json" ]; then
        print_message $RED "错误: 未找到config.json配置文件"
        exit 1
    fi
    
    # 检查API配置
    if python3 -c "
import json
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    api_key = config.get('api', {}).get('api_key', '')
    if not api_key or api_key == 'your-api-key-here':
        print('WARNING: API密钥未配置')
    else:
        print('API配置正常')
except Exception as e:
    print(f'ERROR: 配置文件格式错误: {e}')
    exit(1)
"; then
        print_message $GREEN "配置文件检查通过"
    else
        print_message $YELLOW "警告: 配置文件可能存在问题"
    fi
}

# 创建必要目录
create_directories() {
    print_message $BLUE "创建必要目录..."
    
    directories=(
        "data/input"
        "data/output" 
        "data/pdfs"
        "data/texts"
        "data/retrieved"
        "data/cleaned"
        "data/qa_results"
        "data/quality_checked"
        "data/rewritten"
        "data/final_output"
        "temp"
        "logs"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        print_message $GREEN "创建目录: $dir"
    done
}

# 启动本地模型服务
start_local_models() {
    print_message $BLUE "启动本地模型服务..."
    
    # 检查Ollama是否已安装
    if command -v ollama &> /dev/null; then
        print_message $YELLOW "启动Ollama服务..."
        
        # 检查Ollama是否已运行
        if ! pgrep -f "ollama serve" > /dev/null; then
            nohup ollama serve > logs/ollama.log 2>&1 &
            sleep 3
            print_message $GREEN "Ollama服务已启动"
        else
            print_message $YELLOW "Ollama服务已在运行"
        fi
        
        # 检查并下载模型
        print_message $YELLOW "检查Ollama模型..."
        if ! ollama list | grep -q "qwen:7b"; then
            print_message $YELLOW "下载qwen:7b模型..."
            ollama pull qwen:7b
        fi
        
        print_message $GREEN "本地模型服务准备完成"
    else
        print_message $YELLOW "警告: 未安装Ollama，将使用云端API"
    fi
}

# 运行文本召回
run_text_retrieval() {
    local domain=${1:-"semiconductor"}
    local input_path=${2:-"data/pdfs"}
    
    print_message $BLUE "运行文本召回 - 领域: $domain"
    
    python3 run_pipeline.py \
        --mode text_retrieval \
        --input_path "$input_path" \
        --domain "$domain" \
        --output_path "data/retrieved" || {
        print_message $RED "文本召回失败"
        return 1
    }
    
    print_message $GREEN "文本召回完成"
}

# 运行数据清理
run_data_cleaning() {
    local input_path=${1:-"data/retrieved/total_response.pkl"}
    
    print_message $BLUE "运行数据清理"
    
    python3 run_pipeline.py \
        --mode data_cleaning \
        --input_path "$input_path" \
        --output_path "data/cleaned" || {
        print_message $RED "数据清理失败"
        return 1
    }
    
    print_message $GREEN "数据清理完成"
}

# 运行QA生成
run_qa_generation() {
    local domain=${1:-"semiconductor"}
    local input_path=${2:-"data/cleaned/total_response.json"}
    local multimodal=${3:-false}
    
    print_message $BLUE "运行QA生成 - 领域: $domain, 多模态: $multimodal"
    
    local cmd="python3 run_pipeline.py \
        --mode qa_generation \
        --input_path \"$input_path\" \
        --domain \"$domain\" \
        --output_path \"data/qa_results\""
    
    if [ "$multimodal" = true ]; then
        cmd="$cmd --enable_multimodal"
    fi
    
    eval $cmd || {
        print_message $RED "QA生成失败"
        return 1
    }
    
    print_message $GREEN "QA生成完成"
}

# 运行质量控制
run_quality_control() {
    local input_path=${1:-"data/qa_results/results_343.json"}
    local threshold=${2:-"0.7"}
    
    print_message $BLUE "运行质量控制 - 阈值: $threshold"
    
    python3 run_pipeline.py \
        --mode quality_control \
        --input_path "$input_path" \
        --quality_threshold "$threshold" \
        --output_path "data/quality_checked" || {
        print_message $RED "质量控制失败"
        return 1
    }
    
    print_message $GREEN "质量控制完成"
}

# 运行数据改写
run_data_rewriting() {
    local input_path=${1:-"data/quality_checked/final_qa.json"}
    local ratio=${2:-"0.3"}
    local template_type=${3:-"professional"}
    
    print_message $BLUE "运行数据改写 - 比例: $ratio, 模板: $template_type"
    
    python3 model_rewrite/data_generation.py \
        --data-path "$input_path" \
        --output-path "data/rewritten/enhanced_qa.json" \
        --template-type "$template_type" \
        --rewrite-ratio "$ratio" || {
        print_message $RED "数据改写失败"
        return 1
    }
    
    print_message $GREEN "数据改写完成"
}

# 运行完整流水线
run_full_pipeline() {
    local domain=${1:-"semiconductor"}
    local multimodal=${2:-false}
    local input_path=${3:-"data/pdfs"}
    
    print_message $PURPLE "开始完整流水线处理..."
    print_message $CYAN "领域: $domain, 多模态: $multimodal, 输入: $input_path"
    
    # 步骤1: 文本召回
    print_message $YELLOW "步骤 1/6: 文本召回"
    run_text_retrieval "$domain" "$input_path" || return 1
    
    # 步骤2: 数据清理
    print_message $YELLOW "步骤 2/6: 数据清理"
    run_data_cleaning || return 1
    
    # 步骤3: QA生成
    print_message $YELLOW "步骤 3/6: QA生成"
    run_qa_generation "$domain" "data/cleaned/total_response.json" "$multimodal" || return 1
    
    # 步骤4: 质量控制
    print_message $YELLOW "步骤 4/6: 质量控制"
    run_quality_control || return 1
    
    # 步骤5: 数据改写
    print_message $YELLOW "步骤 5/6: 数据改写"
    run_data_rewriting || return 1
    
    # 步骤6: 最终处理
    print_message $YELLOW "步骤 6/6: 最终处理"
    python3 run_pipeline.py \
        --mode final_processing \
        --input_path "data/rewritten/enhanced_qa.json" \
        --output_path "data/final_output" || {
        print_message $RED "最终处理失败"
        return 1
    }
    
    print_message $GREEN "完整流水线处理完成！"
    print_message $CYAN "输出文件位于: data/final_output/"
}

# 创建演示数据
create_demo_data() {
    print_message $BLUE "创建演示数据..."
    
    # 创建示例PDF目录
    mkdir -p data/pdfs/semiconductor
    
    # 创建示例文本文件
    cat > data/texts/sample_semiconductor.txt << 'EOF'
IGZO（铟镓锌氧化物）是一种新型的氧化物半导体材料，具有以下特点：

1. 高载流子迁移率：IGZO材料的电子迁移率可达10-50 cm²/V·s，远高于非晶硅的迁移率。

2. 良好的均匀性：IGZO薄膜在大面积基板上具有良好的均匀性，适合制造大尺寸显示器。

3. 低温工艺：IGZO可以在相对较低的温度下进行加工，有利于在塑料基板上制造柔性显示器。

4. 透明性：IGZO在可见光范围内具有高透明度，适合制造透明电子器件。

5. 稳定性：IGZO材料具有良好的化学稳定性和电学稳定性。

在TFT-LCD和OLED显示器中，IGZO TFT作为像素开关，具有以下优势：
- 低功耗：由于高迁移率，可以使用较小的晶体管尺寸
- 高开关速度：适合高分辨率和高刷新率显示
- 良好的关态电流特性：有利于提高对比度

IGZO技术的发展推动了高分辨率、低功耗显示技术的进步，特别是在移动设备和大尺寸电视领域有重要应用。
EOF
    
    # 创建示例光学文本
    cat > data/texts/sample_optics.txt << 'EOF'
光谱分析是研究物质与电磁辐射相互作用的重要方法：

1. 吸收光谱：
   - 当光通过物质时，特定波长的光被吸收
   - 吸收峰的位置和强度反映物质的组成和浓度
   - Beer-Lambert定律：A = εcl

2. 发射光谱：
   - 物质受激发后发出特征光谱
   - 原子发射光谱具有线状特征
   - 分子发射光谱通常为带状

3. 拉曼光谱：
   - 基于拉曼散射效应
   - 提供分子振动和转动信息
   - 适合研究分子结构和化学键

4. 红外光谱：
   - 波长范围：2.5-25μm
   - 反映分子振动模式
   - 用于化合物鉴定和结构分析

光谱技术在材料科学、化学分析、生物医学等领域有广泛应用。
EOF
    
    print_message $GREEN "演示数据创建完成"
}

# 运行演示
run_demo() {
    print_message $PURPLE "运行完整演示流水线..."
    
    # 创建演示数据
    create_demo_data
    
    print_message $CYAN "演示场景1: 半导体领域QA生成"
    run_full_pipeline "semiconductor" false "data/texts" || return 1
    
    print_message $CYAN "演示场景2: 光学领域QA生成"
    run_full_pipeline "optics" false "data/texts" || return 1
    
    print_message $GREEN "演示完成！请查看 data/final_output/ 目录中的结果"
}

# 显示系统状态
show_status() {
    print_message $BLUE "系统状态检查..."
    
    # 检查Python
    python_version=$(python3 --version 2>/dev/null || echo "未安装")
    print_message $WHITE "Python: $python_version"
    
    # 检查Ollama
    if command -v ollama &> /dev/null; then
        if pgrep -f "ollama serve" > /dev/null; then
            ollama_status="运行中"
        else
            ollama_status="已安装但未运行"
        fi
    else
        ollama_status="未安装"
    fi
    print_message $WHITE "Ollama: $ollama_status"
    
    # 检查配置文件
    if [ -f "config.json" ]; then
        config_status="存在"
    else
        config_status="缺失"
    fi
    print_message $WHITE "配置文件: $config_status"
    
    # 检查目录结构
    if [ -d "data" ] && [ -d "logs" ]; then
        dirs_status="完整"
    else
        dirs_status="不完整"
    fi
    print_message $WHITE "目录结构: $dirs_status"
    
    # 显示磁盘空间
    disk_usage=$(df -h . | awk 'NR==2 {print $4}')
    print_message $WHITE "可用磁盘空间: $disk_usage"
}

# 主函数
main() {
    print_header
    
    # 解析命令行参数
    DOMAIN="semiconductor"
    MULTIMODAL=false
    INPUT_PATH="data/pdfs"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --demo)
                check_python
                create_directories
                run_demo
                exit 0
                ;;
            --install)
                check_python
                install_dependencies
                exit 0
                ;;
            --check)
                show_status
                check_config
                exit 0
                ;;
            --local)
                start_local_models
                exit 0
                ;;
            --pipeline)
                shift
                check_python
                create_directories
                run_full_pipeline "$DOMAIN" "$MULTIMODAL" "$INPUT_PATH"
                exit 0
                ;;
            --text-retrieval)
                shift
                check_python
                create_directories
                run_text_retrieval "$DOMAIN" "$INPUT_PATH"
                exit 0
                ;;
            --qa-generation)
                shift
                check_python
                create_directories
                run_qa_generation "$DOMAIN" "data/cleaned/total_response.json" "$MULTIMODAL"
                exit 0
                ;;
            --quality-control)
                shift
                check_python
                create_directories
                run_quality_control
                exit 0
                ;;
            --data-rewriting)
                shift
                check_python
                create_directories
                run_data_rewriting
                exit 0
                ;;
            --multimodal)
                MULTIMODAL=true
                ;;
            --domain)
                shift
                DOMAIN="$1"
                ;;
            --input-path)
                shift
                INPUT_PATH="$1"
                ;;
            --help|-h)
                print_usage
                exit 0
                ;;
            *)
                print_message $RED "未知选项: $1"
                print_usage
                exit 1
                ;;
        esac
        shift
    done
    
    # 默认显示帮助信息
    print_usage
}

# 执行主函数
main "$@"