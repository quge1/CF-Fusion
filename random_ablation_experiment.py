#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
随机策略消融实验（Random Ablation Experiment）
对比动态策略 vs 随机策略的性能差异

核心思想：
- 复用 pure_llm_cache 中的 LLM 判断结果
- 随机选择相同数量的样本调用 LLM（与动态策略调用率相同）
- 零 API 调用，快速实验
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

# 目录配置
RESULTS_DIR = "results"
PURE_LLM_CACHE_DIR = "pure_llm_cache"
ABLATION_OUTPUT_DIR = "ablation_study_v2/random_vs_dynamic"


def ensure_directories():
    """确保输出目录存在"""
    os.makedirs(ABLATION_OUTPUT_DIR, exist_ok=True)


def load_dynamic_results(smell_type: str, fold: int) -> Dict:
    """加载动态策略（CF框架）的测试结果"""
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
    """
    从 pure_llm_cache 中查找 LLM 判断结果
    返回: (prediction, judgment_text)
    """
    if entity_key in pure_cache:
        judgment = pure_cache[entity_key].get('llm_judgment', '')
        # 转换为二进制预测
        if judgment == '是':
            return 1, judgment
        else:
            return 0, judgment
    return None, None


def calculate_metrics(y_true: List[int], y_pred: List[int], y_prob: List[float] = None) -> Dict:
    """计算性能指标"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # 计算 AUC
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
    """
    运行随机策略消融实验
    
    步骤：
    1. 加载动态策略结果，统计 LLM 调用数量
    2. 随机选择相同数量的样本
    3. 从 pure_llm_cache 获取 LLM 判断
    4. 计算随机策略的性能指标
    """
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
    random.seed(RANDOM_SEED + fold)  # 每个 fold 使用不同的种子
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
        
        # 判断是否选中调用 LLM
        if i in random_selected_set:
            # 从 pure_llm_cache 获取 LLM 判断
            llm_pred, llm_judgment = find_llm_judgment(entity_key, pure_cache)
            
            if llm_pred is not None:
                # 缓存命中
                cache_hits += 1
                final_prediction = llm_pred
                final_confidence = 0.7 if llm_pred == 1 else 0.3  # 估算置信度
                llm_enhanced = True
                decision_strategy = "Random LLM Ablation: 选中调用LLM (缓存命中)"
            else:
                # 缓存未命中，使用 DNN 结果
                cache_misses += 1
                final_prediction = dnn_prediction
                final_confidence = dnn_confidence
                llm_enhanced = False
                decision_strategy = "Random LLM Ablation: 选中调用LLM (缓存未命中，使用DNN)"
                llm_judgment = ""
        else:
            # 未选中，使用 DNN 结果
            final_prediction = dnn_prediction
            final_confidence = dnn_confidence
            llm_enhanced = False
            decision_strategy = "Random LLM Ablation: 未选中调用LLM"
            llm_judgment = ""
        
        random_results.append({
            'entity_key': entity_key,
            'smell_type': smell_type,
            'dnn_confidence': dnn_confidence,
            'dnn_prediction': dnn_prediction,
            'final_confidence': final_confidence,
            'final_prediction': final_prediction,
            'decision_strategy': decision_strategy,
            'was_corrected': final_prediction != dnn_prediction,
            'true_label': true_label,
            'llm_enhanced': llm_enhanced,
            'llm_judgment': llm_judgment,
            'llm_reason': '',
            'llm_suggested_confidence': None
        })
    
    # 6. 计算随机策略的性能指标
    y_true = [r['true_label'] for r in random_results]
    y_pred = [r['final_prediction'] for r in random_results]
    y_prob = [r['final_confidence'] for r in random_results]
    
    random_metrics = calculate_metrics(y_true, y_pred, y_prob)
    
    # 7. 计算动态策略的性能指标（用于对比）
    dynamic_y_true = [s['true_label'] for s in all_samples]
    dynamic_y_pred = [s['final_prediction'] for s in all_samples]
    dynamic_y_prob = [s.get('final_confidence', s.get('dnn_confidence', 0.5)) for s in all_samples]
    
    dynamic_metrics = calculate_metrics(dynamic_y_true, dynamic_y_pred, dynamic_y_prob)
    
    # 8. 计算提升幅度
    improvement = {
        'accuracy': dynamic_metrics['accuracy'] - random_metrics['accuracy'],
        'precision': dynamic_metrics['precision'] - random_metrics['precision'],
        'recall': dynamic_metrics['recall'] - random_metrics['recall'],
        'f1_score': dynamic_metrics['f1_score'] - random_metrics['f1_score'],
        'auc': dynamic_metrics['auc'] - random_metrics['auc']
    }
    
    # 9. 保存随机策略结果
    output = {
        'smell_type': smell_type,
        'fold': fold,
        'total_samples': total_samples,
        'processed_samples': len(random_results),
        'llm_call_count': dynamic_llm_count,
        'cache_hits': cache_hits,
        'api_calls': 0,  # 零 API 调用
        'original_llm_count': dynamic_llm_count,
        'results': random_results,
        'metrics': random_metrics,
        'dynamic_metrics': dynamic_metrics,
        'improvement': improvement
    }
    
    output_file = f"{ABLATION_OUTPUT_DIR}/{smell_type}_fold{fold}_random_ablation.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"完成 (样本:{total_samples}, LLM:{dynamic_llm_count}, 缓存:{cache_hits}/{cache_hits+cache_misses})")
    
    return output


def generate_summary(all_results: Dict):
    """生成汇总分析 - 五折汇总计算（类似 analyze_cross_validation.py）"""
    print(f"\n{'='*80}")
    print("消融实验汇总分析（五折汇总）")
    print(f"{'='*80}")
    
    summary = {}
    
    for smell_type in ['blob', 'data_class', 'feature_envy', 'long_method']:
        if smell_type not in all_results or not all_results[smell_type]:
            continue
        
        fold_results = all_results[smell_type]
        
        # 汇总五折所有样本（类似 analyze_cross_validation.py 的方式）
        all_dynamic_y_true = []
        all_dynamic_y_pred = []
        all_dynamic_y_prob = []
        all_random_y_true = []
        all_random_y_pred = []
        all_random_y_prob = []
        
        total_samples = 0
        total_llm_calls = 0
        
        for fold_result in fold_results:
            # 随机策略样本（从 fold_result['results'] 获取）
            for r in fold_result['results']:
                all_random_y_true.append(r['true_label'])
                all_random_y_pred.append(r['final_prediction'])
                all_random_y_prob.append(r['final_confidence'])
            
            # 动态策略样本（从原始结果文件加载）
            dynamic_data = load_dynamic_results(smell_type, fold_result['fold'])
            if dynamic_data:
                for s in dynamic_data.get('results', []):
                    all_dynamic_y_true.append(s['true_label'])
                    all_dynamic_y_pred.append(s['final_prediction'])
                    all_dynamic_y_prob.append(s.get('final_confidence', s.get('dnn_confidence', 0.5)))
            
            total_samples += fold_result['total_samples']
            total_llm_calls += fold_result['llm_call_count']
        
        # 计算五折汇总的性能指标
        dynamic_metrics = calculate_metrics(all_dynamic_y_true, all_dynamic_y_pred, all_dynamic_y_prob)
        random_metrics = calculate_metrics(all_random_y_true, all_random_y_pred, all_random_y_prob)
        
        # 计算提升幅度
        improvement = {
            'accuracy': dynamic_metrics['accuracy'] - random_metrics['accuracy'],
            'precision': dynamic_metrics['precision'] - random_metrics['precision'],
            'recall': dynamic_metrics['recall'] - random_metrics['recall'],
            'f1_score': dynamic_metrics['f1_score'] - random_metrics['f1_score'],
            'auc': dynamic_metrics['auc'] - random_metrics['auc']
        }
        
        summary[smell_type] = {
            'smell_type': smell_type,
            'total_samples': total_samples,
            'total_llm_calls': total_llm_calls,
            'llm_call_rate': total_llm_calls / total_samples if total_samples > 0 else 0,
            'fold_count': len(fold_results),
            'dynamic_metrics': dynamic_metrics,
            'random_metrics': random_metrics,
            'improvement': improvement,
            'fold_results': fold_results
        }
        
        print(f"\n{'='*60}")
        print(f"坏味类型: {smell_type}")
        print(f"{'='*60}")
        print(f"总样本数: {total_samples}")
        print(f"LLM 调用数: {total_llm_calls} ({summary[smell_type]['llm_call_rate']:.1%})")
        print(f"Fold 数: {len(fold_results)}")
        
        print(f"\n动态策略性能指标:")
        print(f"   Accuracy:  {dynamic_metrics['accuracy']:.4f}")
        print(f"   Precision: {dynamic_metrics['precision']:.4f}")
        print(f"   Recall:    {dynamic_metrics['recall']:.4f}")
        print(f"   F1-Score:  {dynamic_metrics['f1_score']:.4f}")
        print(f"   AUC:       {dynamic_metrics['auc']:.4f}")
        
        print(f"\n随机策略性能指标:")
        print(f"   Accuracy:  {random_metrics['accuracy']:.4f}")
        print(f"   Precision: {random_metrics['precision']:.4f}")
        print(f"   Recall:    {random_metrics['recall']:.4f}")
        print(f"   F1-Score:  {random_metrics['f1_score']:.4f}")
        print(f"   AUC:       {random_metrics['auc']:.4f}")
        
        print(f"\n动态策略 vs 随机策略提升:")
        print(f"   ΔAccuracy:  {improvement['accuracy']:+.4f}")
        print(f"   ΔPrecision: {improvement['precision']:+.4f}")
        print(f"   ΔRecall:    {improvement['recall']:+.4f}")
        print(f"   ΔF1-Score:  {improvement['f1_score']:+.4f}")
        print(f"   ΔAUC:       {improvement['auc']:+.4f}")
    
    # 保存汇总
    summary_file = f"{ABLATION_OUTPUT_DIR}/analysis_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*80}")
    print(f"汇总已保存: {summary_file}")
    
    # 生成论文格式表格
    print(f"\n{'='*80}")
    print("论文格式表格（五折汇总）")
    print(f"{'='*80}")
    print(f"{'坏味类型':<15} {'总样本':<10} {'LLM调用':<10} {'动态F1':<10} {'随机F1':<10} {'ΔF1':<10} {'动态AUC':<10} {'随机AUC':<10}")
    print("-" * 80)
    
    for smell_type in ['blob', 'data_class', 'feature_envy', 'long_method']:
        if smell_type not in summary:
            continue
        
        s = summary[smell_type]
        total = s['total_samples']
        llm_calls = s['total_llm_calls']
        dynamic_f1 = s['dynamic_metrics']['f1_score']
        random_f1 = s['random_metrics']['f1_score']
        delta_f1 = s['improvement']['f1_score']
        dynamic_auc = s['dynamic_metrics']['auc']
        random_auc = s['random_metrics']['auc']
        
        print(f"{smell_type:<15} {total:<10} {llm_calls:<10} {dynamic_f1:<10.4f} {random_f1:<10.4f} {delta_f1:+<10.4f} {dynamic_auc:<10.4f} {random_auc:<10.4f}")


def main():
    """主函数"""
    ensure_directories()
    
    print("="*80)
    print("随机策略消融实验（Random Ablation Experiment）")
    print("对比：动态策略 vs 随机策略")
    print("="*80)
    
    smell_types = ['blob', 'data_class', 'feature_envy', 'long_method']
    all_results = {}
    
    for smell_type in smell_types:
        all_results[smell_type] = []
        
        for fold in range(1, 6):
            result = run_random_ablation(smell_type, fold)
            if result:
                all_results[smell_type].append(result)
    
    # 生成汇总
    generate_summary(all_results)
    
    print(f"\n{'='*80}")
    print("消融实验完成！")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
