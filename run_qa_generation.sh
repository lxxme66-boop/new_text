#!/bin/bash

# 半导体QA生成系统运行脚本
# 用法: ./run_qa_generation.sh [options]

# 设置默认值
INPUT_DIR="data/texts"
OUTPUT_DIR="data/output"
MODEL="qwq_32"
BATCH_SIZE=32
GPU_DEVICES="4,5,6,7"
PYTHON_CMD="python3"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 显示帮助信息
show_help() {
    echo "半导体QA生成系统运行脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -i, --input DIR        输入文本目录 (默认: $INPUT_DIR)"
    echo "  -o, --output DIR       输出目录 (默认: $OUTPUT_DIR)"
    echo "  -m, --model MODEL      使用的模型 (默认: $MODEL)"
    echo "  -b, --batch SIZE       批处理大小 (默认: $BATCH_SIZE)"
    echo "  -g, --gpu DEVICES      GPU设备ID (默认: $GPU_DEVICES)"
    echo "  -h, --help             显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                     # 使用默认参数运行"
    echo "  $0 -i /path/to/texts   # 指定输入目录"
    echo "  $0 -g 0,1,2,3          # 指定GPU设备"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -b|--batch)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_DEVICES="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}错误: 未知选项 $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 检查Python环境
echo -e "${YELLOW}检查Python环境...${NC}"
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo -e "${RED}错误: 未找到Python3${NC}"
    exit 1
fi

# 检查必要的目录
echo -e "${YELLOW}检查目录结构...${NC}"
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}错误: 输入目录 $INPUT_DIR 不存在${NC}"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"

# 显示运行参数
echo -e "${GREEN}运行参数:${NC}"
echo "  输入目录: $INPUT_DIR"
echo "  输出目录: $OUTPUT_DIR"
echo "  模型: $MODEL"
echo "  批处理大小: $BATCH_SIZE"
echo "  GPU设备: $GPU_DEVICES"
echo ""

# 运行主程序
echo -e "${YELLOW}开始运行QA生成流程...${NC}"

# 方法1: 使用text_processor作为入口
if [ -f "text_processor.py" ]; then
    echo -e "${GREEN}使用text_processor.py处理文本...${NC}"
    $PYTHON_CMD text_processor.py --input "$INPUT_DIR" --output "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/logs/text_processor.log"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo -e "${RED}文本处理失败${NC}"
        exit 1
    fi
fi

# 方法2: 直接使用semiconductor_qa_generator
if [ -f "semiconductor_qa_generator.py" ]; then
    echo -e "${GREEN}运行核心QA生成器...${NC}"
    
    # 创建临时Python脚本来运行异步函数
    cat > "$OUTPUT_DIR/run_qa_gen.py" << EOF
import asyncio
import json
from semiconductor_qa_generator import run_semiconductor_qa_generation

async def main():
    results = await run_semiconductor_qa_generation(
        raw_folders=["$INPUT_DIR"],
        save_paths=["$OUTPUT_DIR/qa_results.json"],
        model_name="$MODEL",
        batch_size=$BATCH_SIZE,
        gpu_devices="$GPU_DEVICES",
        output_dir="$OUTPUT_DIR"
    )
    
    # 保存统计信息
    with open("$OUTPUT_DIR/statistics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("QA生成完成!")
    print(f"结果保存在: $OUTPUT_DIR/qa_results.json")
    print(f"统计信息保存在: $OUTPUT_DIR/statistics.json")

if __name__ == "__main__":
    asyncio.run(main())
EOF

    $PYTHON_CMD "$OUTPUT_DIR/run_qa_gen.py" 2>&1 | tee "$OUTPUT_DIR/logs/qa_generation.log"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo -e "${RED}QA生成失败${NC}"
        exit 1
    fi
    
    # 清理临时文件
    rm -f "$OUTPUT_DIR/run_qa_gen.py"
fi

# 显示结果统计
if [ -f "$OUTPUT_DIR/statistics.json" ]; then
    echo -e "${GREEN}生成统计:${NC}"
    $PYTHON_CMD -c "
import json
with open('$OUTPUT_DIR/statistics.json', 'r') as f:
    stats = json.load(f)
    if 'pipeline_summary' in stats:
        summary = stats['pipeline_summary']
        print(f'  处理文件数: {summary.get(\"total_files_processed\", 0)}')
        print(f'  通过质量评估文件数: {summary.get(\"files_passed_text_quality\", 0)}')
        print(f'  生成问题总数: {summary.get(\"total_questions_generated\", 0)}')
        print(f'  通过质量评估问题数: {summary.get(\"questions_passed_quality\", 0)}')
        print(f'  总处理时间: {summary.get(\"total_processing_time\", 0):.2f}秒')
"
fi

echo -e "${GREEN}QA生成流程完成!${NC}"
echo -e "${GREEN}结果保存在: $OUTPUT_DIR${NC}"