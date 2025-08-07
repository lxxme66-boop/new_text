# 本地模型配置说明

## 问题分析

您遇到的错误是因为程序尝试连接远程API服务器 `http://0.0.0.0:8080/v1`，但该服务器无法访问，导致连接错误。

## 解决方案

### 方案1：使用本地模型（推荐）

我已经修改了配置，启用了本地模型支持。现在您可以使用以下方法之一：

#### 1.1 安装并使用Ollama（推荐）

```bash
# 安装Ollama
cd /workspace/new_text
./setup_ollama.sh

# 或手动安装
curl -fsSL https://ollama.ai/install.sh | sh

# 启动Ollama服务
ollama serve &

# 下载中文模型
ollama pull qwen:7b
```

#### 1.2 使用Mock模式（临时测试）

如果暂时无法安装Ollama，可以使用Mock模式：

```bash
export USE_MOCK_API=true
python run_pipeline.py --input_path data/texts --mode full_pipeline
```

### 方案2：使用命令行参数启用本地模型

```bash
# 使用本地模型运行
python run_pipeline.py --input_path data/texts --use_local_models --mode full_pipeline
```

### 方案3：修改配置文件（已完成）

配置文件 `config.json` 已修改：
- `use_local_models`: 已设置为 `true`
- 本地模型配置已启用Ollama

## 测试配置

运行测试脚本验证配置：

```bash
cd /workspace/new_text
python test_local_model.py
```

## 配置说明

### 本地模型配置（config.json）

```json
"api": {
    "use_local_models": true,  // 启用本地模型
    "local_model_priority": ["ollama", "vllm", "transformers"]
},
"models": {
    "local_models": {
        "ollama": {
            "enabled": true,
            "base_url": "http://localhost:11434",
            "model_name": "qwen:7b",  // 推荐的中文模型
            "timeout": 300
        }
    }
}
```

### 支持的本地模型

1. **Ollama模型**（推荐）
   - qwen:7b - 通义千问7B（中文支持好）
   - qwen:14b - 通义千问14B
   - yi:6b - Yi模型6B
   - llama2:7b-chinese - Llama2中文版

2. **其他选项**
   - VLLM（需要GPU）
   - Transformers（需要较大内存）

## 错误处理

程序现在包含以下错误处理机制：

1. **自动降级**：如果本地模型不可用，会尝试使用API
2. **Mock模式**：用于测试，不需要任何模型
3. **详细日志**：所有错误都会记录在 `logs/` 目录

## 运行示例

### 完整流水线

```bash
# 使用本地模型
python run_pipeline.py --input_path data/texts --domain semiconductor

# 使用Mock模式测试
export USE_MOCK_API=true
python run_pipeline.py --input_path data/texts --domain semiconductor
```

### 单独运行文本召回

```bash
python text_main_batch_inference_enhanced.py --txt_path data/texts --index 43
```

## 常见问题

1. **Ollama服务未启动**
   ```bash
   # 检查服务状态
   curl http://localhost:11434/api/tags
   
   # 启动服务
   ollama serve
   ```

2. **模型未下载**
   ```bash
   # 列出已安装模型
   ollama list
   
   # 下载模型
   ollama pull qwen:7b
   ```

3. **内存不足**
   - 使用较小的模型（如qwen:7b而不是qwen:14b）
   - 减少批处理大小
   - 使用Mock模式进行测试

## 总结

现在您的系统已配置为优先使用本地模型，这样可以避免网络连接问题。如果本地模型不可用，系统会自动提供详细的错误信息和解决建议。