#!/usr/bin/env python3
"""
半导体显示技术领域智能QA生成系统 - 安装测试脚本
用于验证系统安装是否正确，基本功能是否可用
"""

import sys
import os
import json
import importlib
from pathlib import Path

# 颜色定义
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

def print_info(msg):
    print(f"{Colors.BLUE}[信息]{Colors.NC} {msg}")

def print_success(msg):
    print(f"{Colors.GREEN}[成功]{Colors.NC} {msg}")

def print_error(msg):
    print(f"{Colors.RED}[错误]{Colors.NC} {msg}")

def print_warning(msg):
    print(f"{Colors.YELLOW}[警告]{Colors.NC} {msg}")

def check_python_version():
    """检查Python版本"""
    print_info("检查Python版本...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print_success(f"Python版本: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python版本过低: {version.major}.{version.minor}")
        return False

def check_dependencies():
    """检查核心依赖"""
    print_info("检查核心依赖...")
    
    dependencies = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "numpy": "NumPy",
        "pandas": "Pandas",
        "tqdm": "tqdm",
        "aiohttp": "aiohttp",
        "requests": "requests",
        "jieba": "jieba",
        "loguru": "loguru",
        "rich": "rich"
    }
    
    missing = []
    for module, name in dependencies.items():
        try:
            importlib.import_module(module)
            print_success(f"{name} 已安装")
        except ImportError:
            print_error(f"{name} 未安装")
            missing.append(module)
    
    return len(missing) == 0, missing

def check_project_structure():
    """检查项目结构"""
    print_info("检查项目结构...")
    
    required_dirs = [
        "data/texts",
        "data/pdfs",
        "data/output",
        "data/qa_results",
        "TextGeneration",
        "TextQA",
        "LocalModels"
    ]
    
    required_files = [
        "config.json",
        "requirements.txt",
        "run_full_pipeline.sh",
        "run_step_by_step.sh"
    ]
    
    missing_dirs = []
    missing_files = []
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print_success(f"目录存在: {dir_path}")
        else:
            print_warning(f"目录缺失: {dir_path}")
            missing_dirs.append(dir_path)
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print_success(f"文件存在: {file_path}")
        else:
            print_warning(f"文件缺失: {file_path}")
            missing_files.append(file_path)
    
    return len(missing_dirs) == 0 and len(missing_files) == 0

def check_config():
    """检查配置文件"""
    print_info("检查配置文件...")
    
    config_path = "config.json"
    if not os.path.exists(config_path):
        print_error("配置文件不存在")
        return False
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 检查必要的配置项
        required_keys = ["api", "models", "paths", "processing"]
        missing_keys = []
        
        for key in required_keys:
            if key in config:
                print_success(f"配置项存在: {key}")
            else:
                print_warning(f"配置项缺失: {key}")
                missing_keys.append(key)
        
        return len(missing_keys) == 0
    except Exception as e:
        print_error(f"配置文件读取失败: {e}")
        return False

def check_sample_data():
    """检查示例数据"""
    print_info("检查示例数据...")
    
    text_dir = "data/texts"
    if not os.path.exists(text_dir):
        print_warning("文本目录不存在")
        return False
    
    text_files = list(Path(text_dir).glob("*.txt"))
    if text_files:
        print_success(f"找到 {len(text_files)} 个文本文件:")
        for f in text_files[:5]:  # 只显示前5个
            print(f"  - {f.name}")
        return True
    else:
        print_warning("未找到示例文本文件")
        return False

def test_imports():
    """测试核心模块导入"""
    print_info("测试核心模块导入...")
    
    modules_to_test = [
        ("TextGeneration.text_filter", "文本过滤模块"),
        ("TextQA.enhanced_quality_scorer", "质量评分模块"),
        ("LocalModels.local_model_manager", "本地模型管理"),
    ]
    
    failed = []
    for module_name, desc in modules_to_test:
        try:
            importlib.import_module(module_name)
            print_success(f"{desc} 导入成功")
        except Exception as e:
            print_warning(f"{desc} 导入失败: {e}")
            failed.append(module_name)
    
    return len(failed) == 0

def main():
    """主测试函数"""
    print("==========================================")
    print("半导体显示技术领域智能QA生成系统")
    print("安装测试脚本")
    print("==========================================")
    print()
    
    all_passed = True
    
    # 1. Python版本
    if not check_python_version():
        all_passed = False
    print()
    
    # 2. 依赖检查
    deps_ok, missing_deps = check_dependencies()
    if not deps_ok:
        all_passed = False
        print_warning(f"缺失的依赖: {', '.join(missing_deps)}")
        print_info("请运行: pip install -r requirements.txt")
    print()
    
    # 3. 项目结构
    if not check_project_structure():
        all_passed = False
        print_info("请运行: ./setup_environment.sh 创建缺失的目录")
    print()
    
    # 4. 配置文件
    if not check_config():
        all_passed = False
    print()
    
    # 5. 示例数据
    check_sample_data()
    print()
    
    # 6. 模块导入
    if not test_imports():
        print_warning("部分模块导入失败，可能影响功能使用")
    print()
    
    # 总结
    print("==========================================")
    if all_passed:
        print_success("所有测试通过！系统已准备就绪。")
        print_info("下一步:")
        print("1. 将文本文件放入 data/texts/ 目录")
        print("2. 运行: ./run_full_pipeline.sh")
    else:
        print_error("部分测试未通过，请根据提示修复问题。")
        print_info("建议运行: ./setup_environment.sh")
    print("==========================================")

if __name__ == "__main__":
    main()