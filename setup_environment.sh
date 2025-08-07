#!/bin/bash
# 半导体显示技术领域智能QA生成系统 - 环境设置脚本

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印函数
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

echo "=========================================="
echo "半导体显示技术领域智能QA生成系统"
echo "环境设置脚本"
echo "=========================================="

# 检查Python版本
print_info "检查Python环境..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    print_success "Python版本: $PYTHON_VERSION"
    
    # 检查版本是否满足要求
    REQUIRED_VERSION="3.8"
    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then 
        print_success "Python版本满足要求 (>= 3.8)"
    else
        print_error "Python版本过低，需要 >= 3.8"
        exit 1
    fi
else
    print_error "未找到Python3，请先安装Python 3.8或更高版本"
    exit 1
fi

# 创建虚拟环境
print_info "创建虚拟环境..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "虚拟环境创建成功"
else
    print_warning "虚拟环境已存在"
fi

# 激活虚拟环境
print_info "激活虚拟环境..."
source venv/bin/activate

# 升级pip
print_info "升级pip..."
pip install --upgrade pip

# 安装依赖
print_info "安装项目依赖..."
if [ -f "requirements.txt" ]; then
    # 使用国内镜像加速
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    
    if [ $? -eq 0 ]; then
        print_success "依赖安装成功"
    else
        print_warning "部分依赖安装失败，尝试逐个安装核心依赖..."
        
        # 核心依赖列表
        CORE_DEPS=(
            "torch>=1.13.0"
            "transformers>=4.30.0"
            "numpy>=1.21.0"
            "pandas>=1.3.0"
            "tqdm>=4.62.0"
            "aiohttp>=3.8.0"
            "requests>=2.28.0"
            "PyPDF2>=3.0.0"
            "jieba>=0.42.1"
            "loguru>=0.6.0"
            "rich>=12.0.0"
        )
        
        for dep in "${CORE_DEPS[@]}"; do
            print_info "安装 $dep..."
            pip install "$dep" -i https://pypi.tuna.tsinghua.edu.cn/simple
        done
    fi
else
    print_error "未找到requirements.txt文件"
fi

# 创建必要的目录
print_info "创建项目目录结构..."
mkdir -p data/{texts,pdfs,output,qa_results,final_output,cleaned,retrieved,rewritten,quality_checked}
mkdir -p logs
mkdir -p cache

print_success "目录结构创建完成"

# 检查配置文件
print_info "检查配置文件..."
if [ -f "config.json" ]; then
    print_success "配置文件存在"
else
    print_error "配置文件不存在，创建默认配置..."
    cat > config.json << 'EOF'
{
  "api": {
    "use_local_models": false,
    "ark_url": "http://0.0.0.0:8080/v1",
    "api_key": "your-api-key-here",
    "default_backend": "ark"
  },
  "models": {
    "qa_generator_model": {
      "name": "default-model",
      "temperature": 0.7,
      "max_tokens": 2048
    }
  },
  "paths": {
    "text_dir": "data/texts",
    "output_dir": "data/output",
    "qa_output_dir": "data/qa_results"
  },
  "processing": {
    "batch_size": 32,
    "max_concurrent_tasks": 100
  }
}
EOF
    print_warning "已创建默认配置文件，请根据实际情况修改config.json"
fi

# 设置环境变量
print_info "设置环境变量..."
echo "export PYTHONPATH=\"\${PYTHONPATH}:$(pwd)\"" >> venv/bin/activate
print_success "环境变量设置完成"

# 检查GPU（可选）
print_info "检查GPU环境..."
if command -v nvidia-smi &> /dev/null; then
    print_success "检测到GPU:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    print_warning "未检测到GPU，将使用CPU运行"
fi

# 生成激活脚本
cat > activate_env.sh << 'EOF'
#!/bin/bash
# 快速激活环境
source venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
echo "环境已激活！"
echo "使用 deactivate 退出虚拟环境"
EOF
chmod +x activate_env.sh

echo ""
echo "=========================================="
echo "环境设置完成！"
echo ""
echo "使用方法："
echo "1. 激活环境: source activate_env.sh"
echo "2. 运行项目: ./run_full_pipeline.sh"
echo "3. 查看帮助: ./quick_start.sh --help"
echo "=========================================="

# 提示下一步
print_info "建议下一步操作："
echo "1. 将文本文件放入 data/texts/ 目录"
echo "2. 修改 config.json 配置文件"
echo "3. 运行 ./quick_start.sh --demo 查看演示"