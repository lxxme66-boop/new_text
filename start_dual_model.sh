#!/bin/bash
# 双模型启动脚本

# 配置参数
GENERATOR_MODEL_PATH="${1:-/path/to/generation-model}"
EVALUATOR_MODEL_PATH="${2:-/path/to/Skywork-R1V3-38B}"
GENERATOR_PORT="${3:-8000}"
EVALUATOR_PORT="${4:-8001}"
GENERATOR_GPU="${5:-2}"
EVALUATOR_GPU="${6:-4}"

echo "========================================="
echo "双模型评估系统启动脚本"
echo "========================================="
echo "生成模型: $GENERATOR_MODEL_PATH"
echo "评估模型: $EVALUATOR_MODEL_PATH"
echo "生成模型端口: $GENERATOR_PORT"
echo "评估模型端口: $EVALUATOR_PORT"
echo "========================================="

# 检查模型路径
if [ ! -d "$GENERATOR_MODEL_PATH" ]; then
    echo "错误: 生成模型路径不存在: $GENERATOR_MODEL_PATH"
    exit 1
fi

if [ ! -d "$EVALUATOR_MODEL_PATH" ]; then
    echo "错误: 评估模型路径不存在: $EVALUATOR_MODEL_PATH"
    exit 1
fi

# 启动生成模型
echo "启动生成模型服务..."
nohup python -m vllm.entrypoints.openai.api_server \
    --model $GENERATOR_MODEL_PATH \
    --port $GENERATOR_PORT \
    --tensor-parallel-size $GENERATOR_GPU \
    --dtype auto \
    --trust-remote-code \
    --max-model-len 8192 \
    > generator_model.log 2>&1 &

GENERATOR_PID=$!
echo "生成模型PID: $GENERATOR_PID"

# 等待生成模型启动
echo "等待生成模型启动..."
sleep 10

# 检查生成模型是否启动成功
if ! curl -s http://localhost:$GENERATOR_PORT/v1/models > /dev/null; then
    echo "错误: 生成模型启动失败，请检查日志: generator_model.log"
    kill $GENERATOR_PID 2>/dev/null
    exit 1
fi

echo "生成模型启动成功！"

# 启动评估模型
echo "启动评估模型服务..."
nohup python -m vllm.entrypoints.openai.api_server \
    --model $EVALUATOR_MODEL_PATH \
    --port $EVALUATOR_PORT \
    --tensor-parallel-size $EVALUATOR_GPU \
    --dtype auto \
    --trust-remote-code \
    --max-model-len 32768 \
    --limit-mm-per-prompt "image=20" \
    > evaluator_model.log 2>&1 &

EVALUATOR_PID=$!
echo "评估模型PID: $EVALUATOR_PID"

# 等待评估模型启动
echo "等待评估模型启动..."
sleep 20

# 检查评估模型是否启动成功
if ! curl -s http://localhost:$EVALUATOR_PORT/v1/models > /dev/null; then
    echo "错误: 评估模型启动失败，请检查日志: evaluator_model.log"
    kill $GENERATOR_PID 2>/dev/null
    kill $EVALUATOR_PID 2>/dev/null
    exit 1
fi

echo "评估模型启动成功！"

# 保存PID到文件
echo $GENERATOR_PID > generator.pid
echo $EVALUATOR_PID > evaluator.pid

echo "========================================="
echo "所有服务启动成功！"
echo "生成模型: http://localhost:$GENERATOR_PORT"
echo "评估模型: http://localhost:$EVALUATOR_PORT"
echo "========================================="
echo "使用以下命令停止服务:"
echo "  ./stop_dual_model.sh"
echo "========================================="