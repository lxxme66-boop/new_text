#!/bin/bash
# 快速激活环境
source venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
echo "环境已激活！"
echo "使用 deactivate 退出虚拟环境"
