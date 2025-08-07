#!/bin/bash
# 停止双模型服务

echo "停止双模型服务..."

# 读取PID
if [ -f generator.pid ]; then
    GENERATOR_PID=$(cat generator.pid)
    echo "停止生成模型 (PID: $GENERATOR_PID)..."
    kill $GENERATOR_PID 2>/dev/null
    rm generator.pid
else
    echo "未找到生成模型PID文件"
fi

if [ -f evaluator.pid ]; then
    EVALUATOR_PID=$(cat evaluator.pid)
    echo "停止评估模型 (PID: $EVALUATOR_PID)..."
    kill $EVALUATOR_PID 2>/dev/null
    rm evaluator.pid
else
    echo "未找到评估模型PID文件"
fi

echo "服务已停止"