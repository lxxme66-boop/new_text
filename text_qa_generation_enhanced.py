from TextQA.dataargument import get_total_responses, check_data_quality
from TextQA.enhanced_quality_checker import TextQAQualityIntegrator
import asyncio
import json
import pandas as pd
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate QA pairs from text.")
    parser.add_argument("--index", type=int, default=343, help="index for the qa generation task")
    parser.add_argument("--file_path", type=str, 
                        default="/workspace/text_qa_generation/data/output/total_response.json", 
                        help="path to the input json file")
    parser.add_argument("--pool_size", type=int, default=100, help="number of parallel tasks")
    parser.add_argument("--output_file", type=str, 
                        default="/workspace/text_qa_generation/data/output", 
                        help="path to the output folder")
    parser.add_argument("--ark_url", type=str, default="http://0.0.0.0:8080/v1", help="ARK API URL")
    parser.add_argument("--api_key", type=str, default="ae37bba4-73be-4c22-b1a7-c6b1f5ec3a4b", help="ARK API key")
    parser.add_argument("--model", type=str, default="/mnt/storage/models/Skywork/Skywork-R1V3-38B", 
                        help="Model to use for QA generation")
    parser.add_argument("--check_task", type=bool, default=False, help="Whether to check data quality or not")
    parser.add_argument("--check_indexes", type=tuple, default=(40, 37, 38), help="Indexes to check data quality for")
    parser.add_argument("--check_times", type=int, default=9, help="Number of times to check data quality")
    parser.add_argument("--user_stream", default=False, type=bool)
    parser.add_argument("--enhanced_quality", type=bool, default=True, help="Use enhanced quality checking")
    parser.add_argument("--quality_threshold", type=float, default=0.7, help="Quality threshold for enhanced checking")
    
    args = parser.parse_args()
    file_path = args.file_path
    output_file = os.path.join(args.output_file, f"results_{args.index}.json")
    pool_size = args.pool_size
    
    if not args.check_task:
        index = args.index
        stream = args.user_stream
        results = asyncio.run(get_total_responses(index, file_path, pool_size, stream=stream))
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Generated {len(results)} QA pairs, saved to {output_file}")
        
        # Generate statistics
        stats = generate_qa_statistics(results)
        stats_file = output_file.replace('.json', '_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=4)
        print(f"Statistics saved to {stats_file}")
        
    else:
        ark_url = args.ark_url
        api_key = args.api_key
        model = args.model
        check_indexes = args.check_indexes
        check_times = args.check_times
        
        if args.enhanced_quality:
            # 使用增强质量检查
            print("使用增强质量检查...")
            
            # 加载配置
            with open('/workspace/text_qa_generation/config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 更新配置中的API信息
            config['api']['ark_url'] = ark_url
            config['api']['api_key'] = api_key
            config['models']['qa_generator_model']['path'] = model
            config['quality_control']['enhanced_quality_check']['quality_threshold'] = args.quality_threshold
            
            # 初始化增强质量检查器
            integrator = TextQAQualityIntegrator(config)
            
            # 执行增强质量检查
            quality_report = asyncio.run(integrator.enhanced_quality_check(
                qa_file_path=output_file,
                output_dir=args.output_file,
                quality_threshold=args.quality_threshold
            ))
            
            # 打印质量报告
            print("\n=== 增强质量检查报告 ===")
            print(f"总QA对数量: {quality_report['total_qa_pairs']}")
            print(f"通过QA对数量: {quality_report['passed_qa_pairs']}")
            print(f"通过率: {quality_report['pass_rate']:.2%}")
            print(f"是否达到阈值 ({args.quality_threshold:.1%}): {'是' if quality_report['meets_threshold'] else '否'}")
            
            if 'statistics' in quality_report:
                stats = quality_report['statistics']
                print(f"\n统计信息:")
                print(f"平均问题长度: {stats['avg_question_length']:.1f} 字符")
                print(f"平均答案长度: {stats['avg_answer_length']:.1f} 字符")
                
                if stats['question_types_distribution']:
                    print(f"\n问题类型分布:")
                    for q_type, count in stats['question_types_distribution'].items():
                        print(f"  - {q_type}: {count} 个")
            
        else:
            # 使用原有的简化质量检查
            print("使用原有质量检查...")
            asyncio.run(check_data_quality(ark_url, api_key, model, output_file, check_indexes, 
                                         pool_size=pool_size, check_times=check_times, stream=args.user_stream))


def generate_qa_statistics(qa_data):
    """
    Generate statistics about the QA pairs.
    """
    stats = {
        'total_qa_pairs': 0,
        'question_types': {
            'factual': 0,
            'comparison': 0,
            'reasoning': 0,
            'open_ended': 0
        },
        'source_files': {},
        'average_answer_length': 0,
        'average_question_length': 0
    }
    
    total_answer_length = 0
    total_question_length = 0
    
    for item in qa_data:
        if 'qa_pairs' in item:
            qa_pairs = item['qa_pairs']
            stats['total_qa_pairs'] += len(qa_pairs)
            
            # Count by source file
            source = item.get('source_file', 'unknown')
            if source not in stats['source_files']:
                stats['source_files'][source] = 0
            stats['source_files'][source] += len(qa_pairs)
            
            # Count by question type and calculate lengths
            for qa in qa_pairs:
                q_type = qa.get('question_type', 'unknown')
                if q_type in stats['question_types']:
                    stats['question_types'][q_type] += 1
                    
                total_question_length += len(qa.get('question', ''))
                total_answer_length += len(qa.get('answer', ''))
    
    # Calculate averages
    if stats['total_qa_pairs'] > 0:
        stats['average_question_length'] = total_question_length / stats['total_qa_pairs']
        stats['average_answer_length'] = total_answer_length / stats['total_qa_pairs']
    
    # Calculate percentages
    stats['question_type_percentages'] = {}
    for q_type, count in stats['question_types'].items():
        if stats['total_qa_pairs'] > 0:
            percentage = (count / stats['total_qa_pairs']) * 100
            stats['question_type_percentages'][q_type] = f"{percentage:.1f}%"
    
    return stats