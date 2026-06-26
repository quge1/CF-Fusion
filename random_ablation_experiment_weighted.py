#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
随机策略消融实验（使用 results_weighted 目录）
对比动态策略 vs 随机策略的性能差异
"""

import json
import os
import random
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 设置随机种子保证可复现
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 目录配置 - 使用 results_weighted
RESULTS_DIR = "results_weighted"
PURE_LLM_CACHE_DIR = "pure_llm_cache"
ABLATION_OUTPUT_DIR = "ablation_study_weighted/random_vs_dynamic"


def ensure_directories():
    """确保输出目录存在"""
    os.makedirs(ABLATION_OUTPUT_DIR, exist_ok=True)


def load_dynamic_results(smell_type: str, fold: int) -> Dict:
    """加载动态策略（CF框架）的测试结果 - 使用 results_weighted"""
    file_path = f"{RESULTS_DIR}/{smell_type}_fold{fold}_test_results.json"
    if not os.path.exists(file_path):
        print(f"警告: 文件不存在: {file_path}")
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_pure_llm_cache(smell_type: str) -> Dict:
    """加载 Pure LLM 的缓存结果（仅 fold1）"""
    cache_file = f"{PURE_LLM_CACHE_DIR}/{smell_type}_fold1_pure_cache.json"
    if not os.path.exists(cache_file):
        return {}
    
    with open(cache_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_llm_judgment(entity_key: str, pure_cache: Dict) -> Tuple[int, str]:
    """从 pure_llm_cache 中查找 LLM 判断结果"""
    if entity_key in pure_cache:
        judgment = pure_cache[entity_key].get('llm_judgment', '')
        if judgment == '是':
            return 1, judgment
        else:
            return 0, judgment
    return None, None


def calculate_metrics(y_true: List[int], y_pred: List[int], y_prob: List[float] = None) -> Dict:
    """计算性能指标 - 使用加权F1"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    auc = 0.5
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except:
            auc = 0.5
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc': float(auc)
    }


def run_random_ablation(smell_type: str, fold: int) -> Dict:
    """运行随机策略消融实验"""
    print(f"处理 {smell_type} - Fold {fold}...", end=" ")
    
    # 1. 加载动态策略结果
    dynamic_data = load_dynamic_results(smell_type, fold)
    if not dynamic_data:
        print(f"失败")
        return None
    
    all_samples = dynamic_data.get('results', [])
    total_samples = len(all_samples)
    
    # 2. 统计动态策略的 LLM 调用数量
    dynamic_llm_samples = [s for s in all_samples if s.get('llm_enhanced', False)]
    dynamic_llm_count = len(dynamic_llm_samples)
    
    # 3. 加载 Pure LLM 缓存
    pure_cache = load_pure_llm_cache(smell_type)
    
    # 4. 随机选择相同数量的样本
    random.seed(RANDOM_SEED + fold)
    sample_indices = list(range(total_samples))
    random_selected_indices = random.sample(sample_indices, dynamic_llm_count)
    random_selected_set = set(random_selected_indices)
    
    # 5. 模拟随机策略
    random_results = []
    cache_hits = 0
    cache_misses = 0
    
    for i, sample in enumerate(all_samples):
        entity_key = sample.get('entity_key', '')
        true_label = sample.get('true_label', 0)
        dnn_prediction = sample.get('dnn_prediction', 0)
        dnn_confidence = sample.get('dnn_confidence', 0.5)
        
        if i in random_selected_set:
            llm_pred, llm_judgment = find_llm_judgment(entity_key, pure_cache)
            
            if llm_pred is not None:
                cache_hits += 1
                final_prediction = llm_pred
                final_confidence = 0.7 if llm_pred == 1 else 0.3
                llm_enhanced = True
            else:
                cache_misses += 1
                final_prediction = dnn_prediction
                final_confidence = dnn_confidence
                llm_enhanced = False
        else:
            final_prediction = dnn_prediction
            final_confidence = dnn_confidence
            llm_enhanced = False
        
        random_results.append({
            'entity_key': entity_key,
            'true_label': true_label,
            'final_prediction': final_prediction,
            'final_confidence': final_confidence,
            'llm_enhanced': llm_enhanced
        })
    
    # 6. 计算随机策略的性能指标
    y_true = [r['true_label'] for r in random_results]
    y_pred = [r['final_prediction'] for r in random_results]
    y_prob = [r['final_confidence'] for r in random_results]
    
    random_metrics = calculate_metrics(y_true, y_pred, y_prob)
    
    # 7. 计算动态策略的性能指标
    dynamic_y_true = [s['true_label'] for s in all_samples]
    dynamic_y_pred = [s['final_prediction'] for s in all_samples]
    dynamic_y_prob = [s.get('final_confidence', s.get('dnn_confidence', 0.5)) for s in all_samples]
    
    dynamic_metrics = calculate_metrics(dynamic_y_true, dynamic_y_pred, dynamic_y_prob)
    
    print(f"完成 (样本:{total_samples}, LLM:{dynamic_llm_count}, 随机F1:{random_metrics['f1_score']:.4f}, 动态F1:{dynamic_metrics['f1_score']:.4f})")
    
    return {
        'smell_type': smell_type,
        'fold': fold,
        'total_samples': total_samples,
        'llm_call_count': dynamic_llm_count,
        'random_results': random_results,
        'random_metrics': random_metrics,
        'dynamic_metrics': dynamic_metrics
    }


def generate_summary(all_results: Dict):
    """生成汇总分析"""
    print(f"\n{'='*80}")
    print("消融实验汇总分析（使用 results_weighted）")
    print(f"{'='*80}")
    
    smell_names = {
        'blob': '上帝类',
        'data_class': '数据类',
        'feature_envy': '特征依恋',
        'long_method': '长方法'
    }
    
    print(f"\n{'坏味类型':<12} {'总样本':<10} {'LLM调用':<10} {'随机F1':<10} {'动态F1':<10} {'提升':<10}")
    print("-" * 70)
    
    for smell_type in ['blob', 'data_class', 'feature_envy', 'long_method']:
        if smell_type not in all_results or not all_results[smell_type]:
            continue
        
        fold_results = all_results[smell_type]
        
        # 汇总五折所有样本
        all_dynamic_y_true = []
        all_dynamic_y_pred = []
        all_random_y_true = []
        all_random_y_pred = []
        
        total_samples = 0
        total_llm_calls = 0
        
        for fold_result in fold_results:
            # 从fold_result中获取随机策略结果
            for r in fold_result.get('random_results', []):
                all_random_y_true.append(r['true_label'])
                all_random_y_pred.append(r['final_prediction'])
            
            # 加载动态策略结果
            dynamic_data = load_dynamic_results(smell_type, fold_result['fold'])
            if dynamic_data:
                for s in dynamic_data.get('results', []):
                    all_dynamic_y_true.append(s['true_label'])
                    all_dynamic_y_pred.append(s['final_prediction'])
            
            total_samples += fold_result['total_samples']
            total_llm_calls += fold_result['llm_call_count']
        
        # 计算五折汇总的性能指标
        if all_dynamic_y_true and all_random_y_true:
            dynamic_f1 = f1_score(all_dynamic_y_true, all_dynamic_y_pred, average='weighted', zero_division=0)
            random_f1 = f1_score(all_random_y_true, all_random_y_pred, average='weighted', zero_division=0)
            improvement = dynamic_f1 - random_f1
            
            print(f"{smell_names[smell_type]:<12} {total_samples:<10} {total_llm_calls:<10} {random_f1:<10.4f} {dynamic_f1:<10.4f} {improvement:+.4f}")
    
    print("\n" + "="*80)


def main():
    """主函数"""
    ensure_directories()
    
    print("="*80)
    print("随机策略消融实验（使用 results_weighted 目录）")
    print("="*80)
    
    smell_types = ['blob', 'data_class', 'feature_envy', 'long_method']
    all_results = {}
    
    for smell_type in smell_types:
        all_results[smell_type] = []
        
        for fold in range(1, 6):
            result = run_random_ablation(smell_type, fold)
            if result:
                all_results[smell_type].append(result)
                
                # 保存结果
                output_file = f"{ABLATION_OUTPUT_DIR}/{smell_type}_fold{fold}_random_ablation.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
    
    # 生成汇总
    generate_summary(all_results)
    
    print(f"\n{'='*80}")
    print("消融实验完成！")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
