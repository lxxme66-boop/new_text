#!/usr/bin/env python3
"""
测试QA生成功能
"""

import asyncio
import json
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from text_qa_generation_categorized import CategorizedQAGenerator, QuestionType


async def test_qa_generation():
    """测试QA生成功能"""
    print("=" * 60)
    print("测试智能文本QA生成系统")
    print("=" * 60)
    
    # 创建测试配置
    config = {
        "models": {
            "local_models": {
                "vllm": {
                    "enabled": False  # 使用模拟模式进行测试
                }
            }
        }
    }
    
    # 创建生成器
    generator = CategorizedQAGenerator(config)
    
    # 测试数据
    test_contents = [
        "IGZO (Indium-Gallium-Zinc-Oxide) 是一种非晶氧化物半导体材料，具有高载流子迁移率（10-50 cm²/V·s）、良好的均匀性和稳定性。相比传统的a-Si TFT，IGZO TFT具有更高的开关电流比和更低的亚阈值摆幅。",
        
        "TFT的阈值电压漂移是影响显示器件长期稳定性的关键因素。在正偏压温度应力（PBTS）下，氧空位会俘获电子，导致阈值电压正向漂移。而在负偏压照明应力（NBIS）下，光生空穴被界面态俘获，引起阈值电压负向漂移。",
        
        "顶栅结构相比底栅结构的主要优势在于：1）栅极与源漏电极重叠面积更小，寄生电容降低；2）沟道层在栅极形成前不受工艺损伤；3）可以实现自对准工艺，减少光刻步骤。",
        
        "为了提高IGZO TFT的稳定性，可以采用以下方法：1）优化退火工艺，在300-400°C氧气氛围中退火；2）添加钝化层如SiO2或Al2O3；3）采用双栅结构增强栅控能力；4）掺杂稳定元素如Hf或Ta。"
    ]
    
    print(f"\n准备生成 {len(test_contents)} 个问答对...")
    
    # 生成QA对
    qa_pairs = await generator.generate_batch_categorized_qa(
        test_contents,
        batch_size=2
    )
    
    print(f"\n成功生成 {len(qa_pairs)} 个问答对")
    
    # 显示结果
    print("\n" + "=" * 60)
    print("生成的问答对示例：")
    print("=" * 60)
    
    for i, qa in enumerate(qa_pairs[:4], 1):
        print(f"\n【示例 {i}】")
        print(f"类型: {qa['question_type']}")
        print(f"问题: {qa['question']}")
        print(f"答案: {qa['answer'][:200]}..." if len(qa['answer']) > 200 else f"答案: {qa['answer']}")
        print("-" * 40)
    
    # 统计信息
    stats = generator.get_statistics(qa_pairs)
    print("\n" + "=" * 60)
    print("问题类型分布统计：")
    print("=" * 60)
    
    for q_type, ratio in stats['type_ratios'].items():
        expected = stats['expected_ratios'][q_type]
        print(f"{q_type:12s}: {ratio:.1%} (预期: {expected:.1%})")
    
    # 保存结果
    os.makedirs("test_output", exist_ok=True)
    output_file = "test_output/test_qa_results.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=4)
    
    print(f"\n结果已保存到: {output_file}")
    
    # 保存统计信息
    stats_file = "test_output/test_qa_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)
    
    print(f"统计信息已保存到: {stats_file}")


async def test_single_generation():
    """测试单个QA生成"""
    print("\n" + "=" * 60)
    print("测试单个问答对生成")
    print("=" * 60)
    
    config = {
        "models": {
            "local_models": {
                "vllm": {
                    "enabled": False
                }
            }
        }
    }
    
    generator = CategorizedQAGenerator(config)
    
    # 测试每种类型
    test_content = "IGZO TFT具有高迁移率和良好稳定性，但在长时间偏压下会出现阈值电压漂移。这是由于氧空位和界面态的影响。"
    
    for q_type in QuestionType:
        print(f"\n测试 {q_type.value} 类型:")
        qa = await generator.generate_categorized_qa(test_content, force_type=q_type)
        print(f"问题: {qa['question']}")
        print(f"答案: {qa['answer'][:150]}...")


async def main():
    """主测试函数"""
    try:
        # 测试批量生成
        await test_qa_generation()
        
        # 测试单个生成
        await test_single_generation()
        
        print("\n" + "=" * 60)
        print("所有测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())