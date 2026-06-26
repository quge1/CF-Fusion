#!/usr/bin/env python3
"""
分析 API 成本测量结果
生成论文格式的表格
"""

import json
import os

def load_results():
    """加载测量结果"""
    with open('api_cost_results.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_statistics(measurements):
    """计算统计信息"""
    if not measurements:
        return None
    
    prompt_tokens = [m['prompt_tokens'] for m in measurements]
    completion_tokens = [m['completion_tokens'] for m in measurements]
    total_tokens = [m['total_tokens'] for m in measurements]
    latencies = [m['latency'] for m in measurements]
    
    n = len(measurements)
    
    return {
        'sample_count': n,
        'prompt_tokens': {
            'mean': sum(prompt_tokens) / n,
            'std': (sum((x - sum(prompt_tokens)/n)**2 for x in prompt_tokens) / n) ** 0.5,
            'min': min(prompt_tokens),
            'max': max(prompt_tokens)
        },
        'completion_tokens': {
            'mean': sum(completion_tokens) / n,
            'std': (sum((x - sum(completion_tokens)/n)**2 for x in completion_tokens) / n) ** 0.5,
            'min': min(completion_tokens),
            'max': max(completion_tokens)
        },
        'total_tokens': {
            'mean': sum(total_tokens) / n,
            'std': (sum((x - sum(total_tokens)/n)**2 for x in total_tokens) / n) ** 0.5,
            'min': min(total_tokens),
            'max': max(total_tokens)
        },
        'latency': {
            'mean': sum(latencies) / n,
            'std': (sum((x - sum(latencies)/n)**2 for x in latencies) / n) ** 0.5,
            'min': min(latencies),
            'max': max(latencies)
        }
    }

def format_number(value, decimals=1):
    """格式化数字"""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"

def main():
    results = load_results()
    
    print("="*80)
    print("API Token 消耗和延迟测量结果")
    print("="*80)
    print(f"模型: deepseek-ai/DeepSeek-V3.1")
    print(f"总样本数: 48 (每种坏味 12 个)")
    print("="*80)
    
    smell_names = {
        'blob': 'Blob',
        'data_class': 'Data Class',
        'feature_envy': 'Feature Envy',
        'long_method': 'Long Method'
    }
    
    # 计算每个坏味的统计
    all_stats = {}
    for smell_type, measurements in results.items():
        all_stats[smell_type] = calculate_statistics(measurements)
    
    # 打印表格 1: Token 消耗统计
    print("\n" + "="*80)
    print("表 1: 各坏味类型 LLM API Token 消耗统计")
    print("="*80)
    print(f"{'坏味类型':<15} {'样本数':<8} {'Prompt Tokens':<25} {'Completion Tokens':<25}")
    print(f"{'':15} {'':8} {'Mean±Std':<25} {'Mean±Std':<25}")
    print("-"*80)
    
    for smell_type in ['blob', 'data_class', 'feature_envy', 'long_method']:
        stats = all_stats[smell_type]
        name = smell_names[smell_type]
        prompt = f"{format_number(stats['prompt_tokens']['mean'])}±{format_number(stats['prompt_tokens']['std'])}"
        completion = f"{format_number(stats['completion_tokens']['mean'])}±{format_number(stats['completion_tokens']['std'])}"
        print(f"{name:<15} {stats['sample_count']:<8} {prompt:<25} {completion:<25}")
    
    # 计算总体平均
    all_prompts = []
    all_completions = []
    all_totals = []
    all_latencies = []
    
    for smell_type, measurements in results.items():
        for m in measurements:
            all_prompts.append(m['prompt_tokens'])
            all_completions.append(m['completion_tokens'])
            all_totals.append(m['total_tokens'])
            all_latencies.append(m['latency'])
    
    n_total = len(all_prompts)
    avg_prompt = sum(all_prompts) / n_total
    avg_completion = sum(all_completions) / n_total
    avg_total = sum(all_totals) / n_total
    avg_latency = sum(all_latencies) / n_total
    
    print("-"*80)
    print(f"{'总体平均':<15} {n_total:<8} {format_number(avg_prompt):<25} {format_number(avg_completion):<25}")
    print("="*80)
    
    # 打印表格 2: 延迟统计
    print("\n" + "="*80)
    print("表 2: 各坏味类型 LLM API 延迟统计")
    print("="*80)
    print(f"{'坏味类型':<15} {'样本数':<8} {'延迟 (秒)':<40}")
    print(f"{'':15} {'':8} {'Mean±Std':<40}")
    print("-"*80)
    
    for smell_type in ['blob', 'data_class', 'feature_envy', 'long_method']:
        stats = all_stats[smell_type]
        name = smell_names[smell_type]
        latency = f"{format_number(stats['latency']['mean'])}±{format_number(stats['latency']['std'])}"
        print(f"{name:<15} {stats['sample_count']:<8} {latency:<40}")
    
    print("-"*80)
    latency_std = (sum((x - avg_latency)**2 for x in all_latencies) / n_total) ** 0.5
    print(f"{'总体平均':<15} {n_total:<8} {format_number(avg_latency)}±{format_number(latency_std):<40}")
    print("="*80)
    
    # 打印关键数据摘要
    print("\n" + "="*80)
    print("关键数据摘要 (用于论文)")
    print("="*80)
    print(f"\n1. 平均 Prompt Tokens: {format_number(avg_prompt, 0)}")
    print(f"2. 平均 Completion Tokens: {format_number(avg_completion, 0)}")
    print(f"3. 平均 Total Tokens: {format_number(avg_total, 0)}")
    print(f"4. 平均 API 延迟: {format_number(avg_latency, 2)} 秒")
    print(f"5. 平均 API 延迟范围: {format_number(min(all_latencies), 2)} - {format_number(max(all_latencies), 2)} 秒")
    
    # 计算成本估算 (基于 SiliconFlow 定价)
    # DeepSeek-V3.1: 输入 2元/百万tokens, 输出 8元/百万tokens
    input_cost_per_1m = 2.0  # 元
    output_cost_per_1m = 8.0  # 元
    
    cost_per_call_input = (avg_prompt / 1_000_000) * input_cost_per_1m
    cost_per_call_output = (avg_completion / 1_000_000) * output_cost_per_1m
    cost_per_call_total = cost_per_call_input + cost_per_call_output
    
    print(f"\n6. 单次 API 调用成本估算 (SiliconFlow DeepSeek-V3.1):")
    print(f"   - 输入成本: {cost_per_call_input*1000:.4f} 分")
    print(f"   - 输出成本: {cost_per_call_output*1000:.4f} 分")
    print(f"   - 总成本: {cost_per_call_total*1000:.4f} 分 ({cost_per_call_total:.6f} 元)")
    
    print("="*80)

if __name__ == "__main__":
    main()
