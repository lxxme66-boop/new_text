#!/usr/bin/env python3
"""
分类问题生成模块 - 支持四种问题类型
事实型(15%)、比较型(15%)、推理型(50%)、开放型(20%)
"""

import asyncio
import json
import random
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LocalModels.vllm_client import VLLMClient

logger = logging.getLogger(__name__)


class QuestionType(Enum):
    """问题类型枚举"""
    FACTUAL = "factual"      # 事实型
    COMPARISON = "comparison"  # 比较型
    REASONING = "reasoning"    # 推理型
    OPEN_ENDED = "open_ended"  # 开放型


@dataclass
class QuestionTypeConfig:
    """问题类型配置"""
    type: QuestionType
    ratio: float
    prompt_template: str
    examples: List[str]


class CategorizedQAGenerator:
    """分类问题生成器"""
    
    def __init__(self, config: Dict):
        """初始化生成器"""
        self.config = config
        self.question_configs = self._init_question_configs()
        self.vllm_client = None
        
    def _init_question_configs(self) -> Dict[QuestionType, QuestionTypeConfig]:
        """初始化问题类型配置"""
        configs = {
            QuestionType.FACTUAL: QuestionTypeConfig(
                type=QuestionType.FACTUAL,
                ratio=0.15,
                prompt_template="""你是一个半导体和显示技术领域的专家。基于以下内容，生成一个事实型问题和对应的答案。
事实型问题应该询问具体的指标、数值、性能参数、制备工艺等客观信息。

示例问题：
- JDI开发IGO材料的迁移率、PBTS等参数？制备工艺？
- IGZO TFT的典型迁移率范围是多少？
- 氧化物半导体的带隙一般是多少？

内容：{content}

请生成一个类似的事实型问题和详细的答案。
输出格式：
问题：[你的问题]
答案：[详细的答案]""",
                examples=[
                    "JDI开发IGO材料的迁移率、PBTS等参数？制备工艺？",
                    "IGZO TFT的典型迁移率范围是多少？",
                    "氧化物半导体的带隙一般是多少？"
                ]
            ),
            
            QuestionType.COMPARISON: QuestionTypeConfig(
                type=QuestionType.COMPARISON,
                ratio=0.15,
                prompt_template="""你是一个半导体和显示技术领域的专家。基于以下内容，生成一个比较型问题和对应的答案。
比较型问题应该比较不同材料、结构或方案的差异。

示例问题：
- 顶栅结构的IGZO的寄生电容为什么相对于底栅结构的寄生电容要低？
- IGZO与a-Si TFT相比有哪些优势？
- 不同退火温度对IGZO性能的影响有何差异？

内容：{content}

请生成一个类似的比较型问题和详细的答案。
输出格式：
问题：[你的问题]
答案：[详细的答案]""",
                examples=[
                    "顶栅结构的IGZO的寄生电容为什么相对于底栅结构的寄生电容要低？",
                    "IGZO与a-Si TFT相比有哪些优势？",
                    "不同退火温度对IGZO性能的影响有何差异？"
                ]
            ),
            
            QuestionType.REASONING: QuestionTypeConfig(
                type=QuestionType.REASONING,
                ratio=0.50,
                prompt_template="""你是一个半导体和显示技术领域的专家。基于以下内容，生成一个推理型问题和对应的答案。
推理型问题应该探究机制原理、解释某种行为或结果的原因。

示例问题：
- 在IGZO TFT中，环境气氛中的氧气是如何影响TFT的阈值电压的？
- 氧化物半导体TFT可以在制备过程中通过控制氧含量或通过材料元素成分调控氧空位，请问下如果氧化物半导体中氧空位增加，其迁移率一般是如何变化的？为什么会出现这样的结果呢？
- 与传统的IGZO薄膜相比，为什么SiNx覆盖下的IGZO薄膜其电阻率降低，而SiOx覆盖下的IGZO薄膜其电阻率反而升高呢？

内容：{content}

请生成一个类似的推理型问题和详细的答案，要解释清楚原理和机制。
输出格式：
问题：[你的问题]
答案：[详细的答案，包含原理解释]""",
                examples=[
                    "在IGZO TFT中，环境气氛中的氧气是如何影响TFT的阈值电压的？",
                    "氧化物半导体中氧空位增加，其迁移率一般是如何变化的？为什么？",
                    "为什么SiNx覆盖下的IGZO薄膜电阻率降低？"
                ]
            ),
            
            QuestionType.OPEN_ENDED: QuestionTypeConfig(
                type=QuestionType.OPEN_ENDED,
                ratio=0.20,
                prompt_template="""你是一个半导体和显示技术领域的专家。基于以下内容，生成一个开放型问题和对应的答案。
开放型问题应该询问优化建议、改进方法或解决方案。

示例问题：
- 怎么实现短沟道的顶栅氧化物TFT器件且同时避免器件失效？
- 金属氧化物背板在短时间内驱动OLED显示时会出现残影，请问如何在TFT方面改善残影问题？
- 如何优化IGZO TFT的稳定性？

内容：{content}

请生成一个类似的开放型问题和详细的答案，要提供具体的解决方案或改进建议。
输出格式：
问题：[你的问题]
答案：[详细的解决方案或建议]""",
                examples=[
                    "怎么实现短沟道的顶栅氧化物TFT器件且同时避免器件失效？",
                    "如何在TFT方面改善OLED显示的残影问题？",
                    "如何优化IGZO TFT的稳定性？"
                ]
            )
        }
        
        return configs
    
    async def init_vllm_client(self):
        """初始化vLLM客户端"""
        if self.vllm_client is None:
            vllm_config = self.config.get('models', {}).get('local_models', {}).get('vllm', {})
            if vllm_config.get('enabled', False):
                self.vllm_client = VLLMClient(vllm_config)
                await self.vllm_client.start_server()
                logger.info("vLLM客户端初始化成功")
            else:
                logger.warning("vLLM未启用，将使用模拟模式")
    
    def get_question_type_by_ratio(self) -> QuestionType:
        """根据比例随机选择问题类型"""
        rand = random.random()
        cumulative = 0.0
        
        for q_type, config in self.question_configs.items():
            cumulative += config.ratio
            if rand < cumulative:
                return q_type
        
        return QuestionType.REASONING  # 默认返回推理型
    
    async def generate_categorized_qa(self, content: str, force_type: Optional[QuestionType] = None) -> Dict:
        """生成分类的问答对"""
        # 选择问题类型
        q_type = force_type or self.get_question_type_by_ratio()
        config = self.question_configs[q_type]
        
        # 构建prompt
        prompt = config.prompt_template.format(content=content[:2000])  # 限制内容长度
        
        try:
            # 使用vLLM生成
            if self.vllm_client:
                response = await self.vllm_client.generate(
                    prompt,
                    max_tokens=1024,
                    temperature=0.8,
                    top_p=0.9
                )
            else:
                # 模拟生成（用于测试）
                response = self._mock_generate(q_type, content)
            
            # 解析响应
            qa_pair = self._parse_qa_response(response, q_type, content)
            
        except Exception as e:
            logger.error(f"生成问答对失败: {e}")
            # 返回默认结果
            qa_pair = {
                "question_type": q_type.value,
                "question": f"[生成失败] {q_type.value}类型问题",
                "answer": f"[生成失败] 错误: {str(e)}",
                "content": content[:500],
                "error": str(e)
            }
        
        return qa_pair
    
    def _parse_qa_response(self, response: str, q_type: QuestionType, content: str) -> Dict:
        """解析模型响应，提取问题和答案"""
        lines = response.strip().split('\n')
        question = ""
        answer = ""
        
        # 查找问题和答案
        in_answer = False
        for line in lines:
            if line.startswith('问题：') or line.startswith('问题:'):
                question = line.replace('问题：', '').replace('问题:', '').strip()
            elif line.startswith('答案：') or line.startswith('答案:'):
                answer = line.replace('答案：', '').replace('答案:', '').strip()
                in_answer = True
            elif in_answer and line.strip():
                answer += "\n" + line.strip()
        
        # 如果没有找到格式化的问题和答案，尝试其他解析方式
        if not question or not answer:
            parts = response.split('\n\n')
            if len(parts) >= 2:
                question = parts[0].strip()
                answer = '\n'.join(parts[1:]).strip()
            else:
                # 使用整个响应作为答案
                question = f"{q_type.value}类型问题"
                answer = response.strip()
        
        return {
            "question_type": q_type.value,
            "question": question,
            "answer": answer,
            "content": content[:500],  # 保存部分原始内容
            "generated": True
        }
    
    def _mock_generate(self, q_type: QuestionType, content: str) -> str:
        """模拟生成（用于测试）"""
        examples = {
            QuestionType.FACTUAL: """问题：IGZO TFT的典型载流子迁移率是多少？工作电压范围是什么？
答案：IGZO (Indium-Gallium-Zinc-Oxide) TFT的典型载流子迁移率在10-50 cm²/V·s范围内，比传统的a-Si TFT（约0.5-1 cm²/V·s）高出一个数量级。具体数值取决于材料组成比例和制备工艺。工作电压通常在-20V到+20V范围内，阈值电压一般在0-5V之间。""",
            
            QuestionType.COMPARISON: """问题：顶栅结构IGZO TFT与底栅结构相比，在寄生电容方面有什么优势？
答案：顶栅结构的IGZO TFT相比底栅结构具有更低的寄生电容，主要原因是：1）栅极与源漏电极的重叠面积更小；2）栅介质层可以更好地控制，减少边缘效应；3）顶栅结构中，沟道形成后才沉积栅极，避免了工艺过程中的损伤。这使得顶栅器件具有更快的开关速度和更低的功耗。""",
            
            QuestionType.REASONING: """问题：为什么氧化物半导体中氧空位增加会导致迁移率变化？其机理是什么？
答案：氧空位增加对迁移率的影响主要通过以下机制：1）氧空位作为施主，释放自由电子，增加载流子浓度；2）但过多的氧空位会成为散射中心，降低载流子迁移率；3）氧空位还会影响能带结构，改变导带底的态密度分布。因此，适量的氧空位可以提高导电性，但过量会导致迁移率下降，需要精确控制氧含量以优化器件性能。""",
            
            QuestionType.OPEN_ENDED: """问题：如何改善金属氧化物TFT驱动OLED显示时的残影问题？
答案：改善残影问题可以从以下几个方面入手：1）优化TFT的阈值电压稳定性，通过双栅结构或添加钝化层减少电荷俘获；2）采用补偿电路设计，如6T2C或7T1C像素电路，实时补偿阈值电压漂移；3）优化驱动方案，采用反向偏压或周期性复位来释放俘获电荷；4）改进材料体系，如使用IGZTO等更稳定的氧化物半导体；5）优化工艺条件，如退火温度和气氛，减少缺陷态密度。"""
        }
        
        return examples.get(q_type, "问题：测试问题\n答案：测试答案")
    
    async def generate_batch_categorized_qa(self, contents: List[str], 
                                          batch_size: int = 10) -> List[Dict]:
        """批量生成分类的问答对"""
        # 初始化vLLM客户端
        await self.init_vllm_client()
        
        results = []
        
        # 计算每种类型应该生成的数量
        total = len(contents)
        type_counts = {
            QuestionType.FACTUAL: int(total * 0.15),
            QuestionType.COMPARISON: int(total * 0.15),
            QuestionType.REASONING: int(total * 0.50),
            QuestionType.OPEN_ENDED: int(total * 0.20)
        }
        
        # 确保总数匹配
        diff = total - sum(type_counts.values())
        if diff > 0:
            type_counts[QuestionType.REASONING] += diff
        
        # 为每个内容分配问题类型
        type_assignments = []
        for q_type, count in type_counts.items():
            type_assignments.extend([q_type] * count)
        
        # 随机打乱
        random.shuffle(type_assignments)
        
        # 批量生成
        for i in range(0, len(contents), batch_size):
            batch_contents = contents[i:i+batch_size]
            batch_types = type_assignments[i:i+batch_size]
            
            batch_tasks = []
            for content, q_type in zip(batch_contents, batch_types):
                task = self.generate_categorized_qa(content, force_type=q_type)
                batch_tasks.append(task)
            
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
            
            # 打印进度
            progress = min(i + batch_size, len(contents))
            logger.info(f"生成进度: {progress}/{len(contents)} ({progress/len(contents)*100:.1f}%)")
        
        return results
    
    def get_statistics(self, qa_pairs: List[Dict]) -> Dict:
        """统计问题类型分布"""
        type_counts = {q_type.value: 0 for q_type in QuestionType}
        
        for qa in qa_pairs:
            q_type = qa.get("question_type", "unknown")
            if q_type in type_counts:
                type_counts[q_type] += 1
        
        total = len(qa_pairs)
        type_ratios = {
            q_type: count / total if total > 0 else 0
            for q_type, count in type_counts.items()
        }
        
        return {
            "total": total,
            "type_counts": type_counts,
            "type_ratios": type_ratios,
            "expected_ratios": {
                QuestionType.FACTUAL.value: 0.15,
                QuestionType.COMPARISON.value: 0.15,
                QuestionType.REASONING.value: 0.50,
                QuestionType.OPEN_ENDED.value: 0.20
            }
        }


async def main(args):
    """主函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 加载配置
    config_path = args.config if hasattr(args, 'config') else "config_local.json"
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        logger.warning(f"配置文件 {config_path} 不存在，使用默认配置")
        config = {}
    
    # 创建分类生成器
    generator = CategorizedQAGenerator(config)
    
    # 读取输入数据
    if os.path.exists(args.file_path):
        with open(args.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # 创建测试数据
        logger.warning(f"输入文件 {args.file_path} 不存在，使用测试数据")
        data = [
            {"content": "IGZO是一种氧化物半导体材料，具有高迁移率和良好的稳定性。"},
            {"content": "TFT的阈值电压漂移是影响显示质量的重要因素。"},
            {"content": "顶栅结构相比底栅结构具有更低的寄生电容。"}
        ]
    
    # 提取内容
    contents = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and 'content' in item:
                contents.append(item['content'])
            elif isinstance(item, str):
                contents.append(item)
    elif isinstance(data, dict) and 'content' in data:
        contents.append(data['content'])
    
    if not contents:
        logger.error("未能从输入数据中提取到任何内容")
        return
    
    logger.info(f"提取到 {len(contents)} 条内容")
    
    # 生成分类问答对
    qa_pairs = await generator.generate_batch_categorized_qa(
        contents, 
        batch_size=args.pool_size
    )
    
    # 确保输出目录存在
    os.makedirs(args.output_file, exist_ok=True)
    
    # 保存结果
    output_file = os.path.join(args.output_file, f"categorized_results_{args.index}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=4)
    
    logger.info(f"生成 {len(qa_pairs)} 个问答对，保存到 {output_file}")
    
    # 生成统计信息
    stats = generator.get_statistics(qa_pairs)
    stats_file = output_file.replace('.json', '_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)
    
    logger.info(f"统计信息保存到 {stats_file}")
    logger.info(f"问题类型分布: {stats['type_ratios']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="生成分类的问答对")
    parser.add_argument("--index", type=int, default=1, help="任务索引")
    parser.add_argument("--file_path", type=str, 
                        default="data/output/total_response.json", 
                        help="输入文件路径")
    parser.add_argument("--pool_size", type=int, default=10, help="批处理大小")
    parser.add_argument("--output_file", type=str, 
                        default="data/output", 
                        help="输出目录")
    parser.add_argument("--config", type=str, default="config_local.json", help="配置文件路径")
    
    args = parser.parse_args()
    
    asyncio.run(main(args))