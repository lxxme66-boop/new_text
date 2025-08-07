#!/usr/bin/env python3
"""
批量评估系统：支持大规模模型评估任务
"""
import asyncio
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.asyncio import tqdm

class BatchEvaluationSystem:
    """批量评估系统"""
    
    def __init__(self, 
                 generator_url: str,
                 evaluator_url: str,
                 generator_model: str,
                 evaluator_model: str,
                 output_dir: str = "./evaluation_results"):
        self.generator_url = generator_url
        self.evaluator_url = evaluator_url
        self.generator_model = generator_model
        self.evaluator_model = evaluator_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建会话ID
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / self.session_id
        self.session_dir.mkdir(exist_ok=True)
        
    async def generate_response(self, prompt: str, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """生成单个响应"""
        url = f"{self.generator_url}/v1/chat/completions"
        payload = {
            "model": self.generator_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        try:
            async with session.post(url, json=payload) as response:
                result = await response.json()
                return {
                    "success": True,
                    "response": result["choices"][0]["message"]["content"],
                    "usage": result.get("usage", {})
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": None
            }
    
    async def evaluate_response(self, prompt: str, response: str, 
                              criteria: List[str], session: aiohttp.ClientSession) -> Dict[str, Any]:
        """评估单个响应"""
        eval_prompt = self._build_evaluation_prompt(prompt, response, criteria)
        
        url = f"{self.evaluator_url}/v1/chat/completions"
        payload = {
            "model": self.evaluator_model,
            "messages": [{"role": "user", "content": eval_prompt}],
            "temperature": 0.1,
            "max_tokens": 1500
        }
        
        try:
            async with session.post(url, json=payload) as response:
                result = await response.json()
                eval_text = result["choices"][0]["message"]["content"]
                return {
                    "success": True,
                    "evaluation": self._parse_evaluation(eval_text),
                    "raw_evaluation": eval_text
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "evaluation": None
            }
    
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
            import re
            json_match = re.search(r'\{.*\}', eval_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"raw_text": eval_text}
        except Exception as e:
            return {"parse_error": str(e), "raw_text": eval_text}
    
    async def process_single_item(self, item: Dict[str, Any], criteria: List[str], 
                                session: aiohttp.ClientSession) -> Dict[str, Any]:
        """处理单个评估项"""
        prompt = item["prompt"]
        
        # 生成响应
        gen_result = await self.generate_response(prompt, session)
        
        if not gen_result["success"]:
            return {
                "id": item.get("id", "unknown"),
                "prompt": prompt,
                "generation_error": gen_result["error"],
                "success": False
            }
        
        # 评估响应
        eval_result = await self.evaluate_response(
            prompt, 
            gen_result["response"], 
            criteria,
            session
        )
        
        return {
            "id": item.get("id", "unknown"),
            "prompt": prompt,
            "response": gen_result["response"],
            "generation_usage": gen_result.get("usage", {}),
            "evaluation": eval_result["evaluation"] if eval_result["success"] else None,
            "evaluation_error": eval_result.get("error"),
            "success": gen_result["success"] and eval_result["success"],
            "timestamp": datetime.now().isoformat()
        }
    
    async def process_batch(self, 
                          items: List[Dict[str, Any]], 
                          criteria: List[str],
                          max_concurrent: int = 5) -> List[Dict[str, Any]]:
        """批量处理评估任务"""
        results = []
        
        # 创建信号量控制并发
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(item):
            async with semaphore:
                async with aiohttp.ClientSession() as session:
                    return await self.process_single_item(item, criteria, session)
        
        # 使用进度条
        tasks = [process_with_semaphore(item) for item in items]
        
        for result in await tqdm.gather(*tasks, desc="Processing items"):
            results.append(result)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], filename: str = "results.json"):
        """保存评估结果"""
        filepath = self.session_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {filepath}")
        
        # 同时保存为CSV格式
        df = self._results_to_dataframe(results)
        csv_path = self.session_dir / filename.replace(".json", ".csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"CSV已保存到: {csv_path}")
    
    def _results_to_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """将结果转换为DataFrame"""
        rows = []
        for result in results:
            row = {
                "id": result.get("id"),
                "prompt": result.get("prompt"),
                "response_length": len(result.get("response", "")),
                "success": result.get("success"),
                "timestamp": result.get("timestamp")
            }
            
            # 提取评估分数
            if result.get("evaluation") and isinstance(result["evaluation"], dict):
                row["overall_score"] = result["evaluation"].get("overall_score", None)
                
                # 提取各维度分数
                if "scores" in result["evaluation"]:
                    for criterion, details in result["evaluation"]["scores"].items():
                        if isinstance(details, dict):
                            row[f"score_{criterion}"] = details.get("score", None)
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def generate_report(self, results: List[Dict[str, Any]]):
        """生成评估报告"""
        df = self._results_to_dataframe(results)
        
        # 创建报告
        report_path = self.session_dir / "evaluation_report.html"
        
        html_content = f"""
        <html>
        <head>
            <title>模型评估报告 - {self.session_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ background: #f0f0f0; padding: 10px; margin: 10px 0; }}
                .success {{ color: green; }}
                .failure {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
            </style>
        </head>
        <body>
            <h1>模型评估报告</h1>
            <p>会话ID: {self.session_id}</p>
            <p>生成模型: {self.generator_model}</p>
            <p>评估模型: {self.evaluator_model}</p>
            
            <h2>总体统计</h2>
            <div class="metric">
                <p>总评估数: {len(results)}</p>
                <p>成功率: {df['success'].mean():.1%}</p>
                <p>平均总体评分: {df['overall_score'].mean():.2f}/10</p>
                <p>平均响应长度: {df['response_length'].mean():.0f} 字符</p>
            </div>
            
            <h2>详细结果</h2>
            {self._generate_results_table(results)}
            
            <h2>分数分布</h2>
            <img src="score_distribution.png" alt="分数分布图">
        </body>
        </html>
        """
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        # 生成分数分布图
        self._generate_score_plot(df)
        
        print(f"报告已生成: {report_path}")
    
    def _generate_results_table(self, results: List[Dict[str, Any]]) -> str:
        """生成结果表格HTML"""
        rows = []
        for result in results[:20]:  # 只显示前20条
            if result.get("success"):
                score = result.get("evaluation", {}).get("overall_score", "N/A")
                status = '<span class="success">✓</span>'
            else:
                score = "N/A"
                status = '<span class="failure">✗</span>'
            
            rows.append(f"""
            <tr>
                <td>{result.get('id', 'N/A')}</td>
                <td>{result.get('prompt', '')[:50]}...</td>
                <td>{len(result.get('response', ''))}</td>
                <td>{score}</td>
                <td>{status}</td>
            </tr>
            """)
        
        return f"""
        <table>
            <tr>
                <th>ID</th>
                <th>问题</th>
                <th>响应长度</th>
                <th>总体评分</th>
                <th>状态</th>
            </tr>
            {''.join(rows)}
        </table>
        """
    
    def _generate_score_plot(self, df: pd.DataFrame):
        """生成分数分布图"""
        plt.figure(figsize=(10, 6))
        
        # 提取所有分数列
        score_columns = [col for col in df.columns if col.startswith("score_") or col == "overall_score"]
        
        if score_columns:
            # 创建子图
            fig, axes = plt.subplots(1, len(score_columns), figsize=(5*len(score_columns), 5))
            if len(score_columns) == 1:
                axes = [axes]
            
            for ax, col in zip(axes, score_columns):
                scores = df[col].dropna()
                if len(scores) > 0:
                    ax.hist(scores, bins=10, range=(0, 10), alpha=0.7)
                    ax.set_xlabel("分数")
                    ax.set_ylabel("频次")
                    ax.set_title(col.replace("score_", "").replace("_", " ").title())
                    ax.axvline(scores.mean(), color='red', linestyle='dashed', linewidth=2, label=f'平均: {scores.mean():.2f}')
                    ax.legend()
            
            plt.tight_layout()
            plt.savefig(self.session_dir / "score_distribution.png", dpi=150, bbox_inches='tight')
            plt.close()

# 使用示例
async def run_batch_evaluation():
    """运行批量评估示例"""
    
    # 初始化系统
    system = BatchEvaluationSystem(
        generator_url="http://localhost:8000",
        evaluator_url="http://localhost:8001",
        generator_model="Qwen2.5-7B",
        evaluator_model="Skywork-R1V3-38B"
    )
    
    # 准备测试数据
    test_items = [
        {"id": "001", "prompt": "解释什么是机器学习"},
        {"id": "002", "prompt": "如何学习Python编程"},
        {"id": "003", "prompt": "什么是量子计算"},
        {"id": "004", "prompt": "介绍一下人工智能的应用"},
        {"id": "005", "prompt": "解释深度学习和机器学习的区别"}
    ]
    
    # 评估标准
    criteria = [
        "准确性：信息是否正确",
        "完整性：是否充分回答了问题", 
        "清晰度：表达是否清楚易懂",
        "实用性：回答是否有实际价值"
    ]
    
    # 执行批量评估
    print(f"开始批量评估，共 {len(test_items)} 个项目...")
    results = await system.process_batch(test_items, criteria, max_concurrent=3)
    
    # 保存结果
    system.save_results(results)
    
    # 生成报告
    system.generate_report(results)
    
    # 打印摘要
    successful = sum(1 for r in results if r.get("success"))
    print(f"\n评估完成！")
    print(f"成功: {successful}/{len(results)}")
    print(f"结果保存在: {system.session_dir}")

if __name__ == "__main__":
    asyncio.run(run_batch_evaluation())