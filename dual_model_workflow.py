#!/usr/bin/env python3
"""
双模型工作流程：使用一个模型生成，另一个模型评估
"""
import json
import requests
import asyncio
import aiohttp
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    base_url: str
    model_id: str
    
class DualModelPipeline:
    """生成-评估双模型管道"""
    
    def __init__(self, generator_config: ModelConfig, evaluator_config: ModelConfig):
        self.generator = generator_config
        self.evaluator = evaluator_config
        
    async def generate(self, prompt: str, **kwargs) -> str:
        """使用生成模型生成响应"""
        async with aiohttp.ClientSession() as session:
            url = f"{self.generator.base_url}/v1/chat/completions"
            payload = {
                "model": self.generator.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1000)
            }
            
            async with session.post(url, json=payload) as response:
                result = await response.json()
                return result["choices"][0]["message"]["content"]
    
    async def evaluate(self, prompt: str, response: str, criteria: List[str]) -> Dict[str, Any]:
        """使用评估模型评估生成的响应"""
        evaluation_prompt = self._build_evaluation_prompt(prompt, response, criteria)
        
        async with aiohttp.ClientSession() as session:
            url = f"{self.evaluator.base_url}/v1/chat/completions"
            payload = {
                "model": self.evaluator.model_id,
                "messages": [{"role": "user", "content": evaluation_prompt}],
                "temperature": 0.1,  # 评估时使用低温度
                "max_tokens": 1000
            }
            
            async with session.post(url, json=payload) as response:
                result = await response.json()
                eval_text = result["choices"][0]["message"]["content"]
                return self._parse_evaluation(eval_text)
    
    def _build_evaluation_prompt(self, prompt: str, response: str, criteria: List[str]) -> str:
        """构建评估提示词"""
        criteria_str = "\n".join([f"- {c}" for c in criteria])
        
        return f"""请评估以下AI助手的回答质量。

原始问题：{prompt}

AI回答：{response}

请根据以下标准进行评估：
{criteria_str}

请为每个标准打分（1-10分），并给出简短的理由。
最后给出总体评分和总结。

请以JSON格式输出，格式如下：
{{
    "scores": {{
        "criterion_name": {{"score": 8, "reason": "理由说明"}}
    }},
    "overall_score": 8.5,
    "summary": "总体评价"
}}"""
    
    def _parse_evaluation(self, eval_text: str) -> Dict[str, Any]:
        """解析评估结果"""
        try:
            # 尝试提取JSON部分
            import re
            json_match = re.search(r'\{.*\}', eval_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # 如果无法解析，返回原始文本
                return {"raw_evaluation": eval_text}
        except Exception as e:
            return {"error": str(e), "raw_evaluation": eval_text}
    
    async def process_batch(self, prompts: List[str], criteria: List[str]) -> List[Dict[str, Any]]:
        """批量处理：生成并评估"""
        results = []
        
        for prompt in prompts:
            print(f"\n处理: {prompt[:50]}...")
            
            # 生成
            response = await self.generate(prompt)
            print(f"生成完成，长度: {len(response)}")
            
            # 评估
            evaluation = await self.evaluate(prompt, response, criteria)
            print(f"评估完成")
            
            results.append({
                "prompt": prompt,
                "response": response,
                "evaluation": evaluation,
                "timestamp": datetime.now().isoformat()
            })
            
        return results

# 使用示例
async def main():
    # 配置两个模型
    generator = ModelConfig(
        name="Qwen2.5-7B",
        base_url="http://localhost:8000",
        model_id="Qwen2.5-7B"
    )
    
    evaluator = ModelConfig(
        name="Skywork-R1V3-38B",
        base_url="http://localhost:8001", 
        model_id="Skywork-R1V3-38B"
    )
    
    # 创建管道
    pipeline = DualModelPipeline(generator, evaluator)
    
    # 测试用例
    test_prompts = [
        "请解释什么是机器学习",
        "写一个Python函数来计算斐波那契数列",
        "分析全球气候变化的主要原因"
    ]
    
    # 评估标准
    criteria = [
        "准确性：信息是否正确",
        "完整性：是否充分回答了问题",
        "清晰度：表达是否清楚易懂",
        "实用性：回答是否有实际价值"
    ]
    
    # 执行批量处理
    results = await pipeline.process_batch(test_prompts, criteria)
    
    # 保存结果
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 生成报告
    generate_report(results)

def generate_report(results: List[Dict[str, Any]]):
    """生成评估报告"""
    print("\n" + "="*50)
    print("评估报告")
    print("="*50)
    
    for i, result in enumerate(results, 1):
        print(f"\n### 测试 {i}")
        print(f"问题: {result['prompt'][:100]}...")
        print(f"回答长度: {len(result['response'])} 字符")
        
        if "overall_score" in result["evaluation"]:
            print(f"总体评分: {result['evaluation']['overall_score']}/10")
            
            if "scores" in result["evaluation"]:
                print("\n详细评分:")
                for criterion, details in result["evaluation"]["scores"].items():
                    print(f"  - {criterion}: {details['score']}/10")
                    print(f"    理由: {details['reason']}")
        else:
            print("评估结果:", result["evaluation"])
    
    # 生成汇总统计
    if any("overall_score" in r["evaluation"] for r in results):
        scores = [r["evaluation"]["overall_score"] for r in results 
                 if "overall_score" in r["evaluation"]]
        print(f"\n平均总体评分: {sum(scores)/len(scores):.2f}/10")

if __name__ == "__main__":
    asyncio.run(main())