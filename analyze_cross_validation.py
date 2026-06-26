#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
交叉验证结果分析工具
专门用于分析五折交叉验证的测试结果
"""

import json
import numpy as np
import os
from typing import Dict, List, Set
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 小样本学习标杆案例 - 需要在结果分析时排除
FEW_SHOT_EXCLUSIONS = {
    "blob": [
        "class__blob__none__0__12957__d6555bf5b0b62aef92be79f5f2fbe00426ebee36__C__3__20",
        "class__blob__minor__1__12883__4deb681aaaa79c248115037fc8e399c9876619fd__DefaultHotSpotLoweringProvider__184__809",
        "class__blob__major__1__4028__ac1e6e4035f9307b871478ed47246cf92cfd5f7f__ApplicationResource__63__563",
        "class__blob__critical__1__1096__a35e3a450b4c0134cb097b9e7de76dca08eb6654__FormatParser__22__1683"
    ],
    "data_class": [
        "class__data_class__none__0__12957__d6555bf5b0b62aef92be79f5f2fbe00426ebee36__C__3__20",
        "class__data_class__minor__1__12883__4deb681aaaa79c248115037fc8e399c9876619fd__DefaultHotSpotLoweringProvider__184__809",
        "class__data_class__major__1__4028__ac1e6e4035f9307b871478ed47246cf92cfd5f7f__ApplicationResource__63__563",
        "class__data_class__critical__1__1096__a35e3a450b4c0134cb097b9e7de76dca08eb6654__FormatParser__22__1683"
    ],
    "feature_envy": [
        "method__feature_envy__none__0__12957__d6555bf5b0b62aef92be79f5f2fbe00426ebee36__C__3__20",
        "method__feature_envy__minor__1__12883__4deb681aaaa79c248115037fc8e399c9876619fd__DefaultHotSpotLoweringProvider__184__809",
        "method__feature_envy__major__1__4028__ac1e6e4035f9307b871478ed47246cf92cfd5f7f__ApplicationResource__63__563",
        "method__feature_envy__critical__1__1096__a35e3a450b4c0134cb097b9e7de76dca08eb6654__FormatParser__22__1683"
    ],
    "long_method": [
        "method__long_method__none__0__12957__d6555bf5b0b62aef92be79f5f2fbe00426ebee36__C__3__20",
        "method__long_method__minor__1__12883__4deb681aaaa79c248115037fc8e399c9876619fd__DefaultHotSpotLoweringProvider__184__809",
        "method__long_method__major__1__4028__ac1e6e4035f9307b871478ed47246cf92cfd5f7f__ApplicationResource__63__563",
        "method__long_method__critical__1__1096__a35e3a450b4c0134cb097b9e7de76dca08eb6654__FormatParser__22__1683"
    ]
}

def get_excluded_keys(smell_type: str) -> Set[str]:
    """获取需要排除的标杆案例集合"""
    return set(FEW_SHOT_EXCLUSIONS.get(smell_type, []))

def filter_results_by_exclusions(results_data: List[Dict], smell_type: str) -> tuple:
    """过滤结果数据，排除标杆案例（用于结果分析）"""
    excluded_keys = get_excluded_keys(smell_type)
    if not excluded_keys:
        return results_data, 0
    
    filtered_data = []
    excluded_count = 0
    
    for item in results_data:
        entity_key = item.get('entity_key', '')
        if entity_key in excluded_keys:
            excluded_count += 1
            continue
        filtered_data.append(item)
    
    if excluded_count > 0:
        print(f"🚫 结果分析: 排除 {excluded_count} 个标杆案例")
        print(f"📊 剩余样本: {len(filtered_data)}")
    
    return filtered_data, excluded_count

def load_results(file_path):
    """加载结果文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 加载结果文件失败: {e}")
        return None

def calculate_metrics(y_true, y_pred, y_prob=None):
    """计算性能指标"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 计算基础指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    # 计算AUC
    auc = 0.5
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            if len(np.unique(y_true)) == 2:  # 二分类
                auc = roc_auc_score(y_true, y_prob)
            else:  # 多分类
                auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except Exception as e:
            print(f"⚠️ AUC计算失败: {e}")
            auc = 0.5
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc': float(auc)
    }

def analyze_fold_results(smell_type, fold_idx):
    """分析单个fold的结果"""
    file_path = f'results/{smell_type}_fold{fold_idx}_test_results.json'
    #file_path = f'ablation_study_v2/random_vs_dynamic/{smell_type}_fold{fold_idx}_random_ablation.json'
    print(f"📊 分析 {smell_type} - Fold {fold_idx}")
    print(f"📁 文件路径: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ 结果文件不存在: {file_path}")
        return None
    
    data = load_results(file_path)
    if not data:
        return None
    
    # 获取实际的results数据
    if isinstance(data, dict) and 'results' in data:
        results_data = data['results']
    else:
        results_data = data
    
    print(f"📈 总样本数: {len(results_data)}")
    
    # 排除标杆案例，避免LLM数据泄露影响结果
    results_data, excluded_count = filter_results_by_exclusions(results_data, smell_type)
    
    # 提取有效数据
    valid_data = []
    for item in results_data:
        if 'true_label' in item and 'dnn_prediction' in item and 'final_prediction' in item:
            valid_data.append(item)
    
    if not valid_data:
        print("❌ 没有有效的数据")
        return None
    
    print(f"✅ 有效样本数: {len(valid_data)}")
    
    # 提取标签和预测结果
    true_labels = [item['true_label'] for item in valid_data]
    dnn_predictions = [item['dnn_prediction'] for item in valid_data]
    final_predictions = [item['final_prediction'] for item in valid_data]
    
    # 提取置信度
    dnn_confidences = [item.get('dnn_confidence', 0.5) for item in valid_data]
    final_confidences = [item.get('final_confidence', item.get('dnn_confidence', 0.5)) for item in valid_data]
    
    # 计算类别分布
    unique, counts = np.unique(true_labels, return_counts=True)
    class_distribution = {str(k): int(v) for k, v in zip(unique, counts)}
    print(f"📊 类别分布: {class_distribution}")
    
    # 计算DNN原始性能
    dnn_metrics = calculate_metrics(true_labels, dnn_predictions, dnn_confidences)
    
    # 计算LLM辅助后性能
    llm_metrics = calculate_metrics(true_labels, final_predictions, final_confidences)
    
    # 计算改进幅度
    improvements = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc']:
        # 始终添加该指标，即使DNN指标为0
        dnn_value = dnn_metrics[metric]
        llm_value = llm_metrics[metric]
        improvement = llm_value - dnn_value
        # 避免除以0
        if dnn_value != 0:
            percent_improve = (improvement / abs(dnn_value)) * 100
        else:
            # 如果DNN为0但LLM不为0，视为100%改进；如果都为0，视为0%改进
            percent_improve = 100.0 if llm_value != 0 else 0.0
        improvements[metric] = {
            'absolute': improvement,
            'percent': percent_improve
        }
    
    return {
        'fold_idx': fold_idx,
        'file_path': file_path,
        'dataset_info': {
            'total_samples': len(results_data),
            'valid_samples': len(valid_data),
            'class_distribution': class_distribution
        },
        'dnn_original': dnn_metrics,
        'llm_enhanced': llm_metrics,
        'improvements': improvements
    }

def analyze_cross_validation(smell_type, selected_folds=None):
    """分析交叉验证结果"""
    if selected_folds is None:
        # 自动检测存在的fold
        selected_folds = []
        for fold_idx in range(1, 6):  # 假设最多5个fold
            file_path = f'results/{smell_type}_fold{fold_idx}_test_results.json'
            if os.path.exists(file_path):
                selected_folds.append(fold_idx)
    
    print("🚀" + "="*60)
    print(f"📊 交叉验证结果分析")
    print(f"📊 坏味类型: {smell_type}")
    print(f"📊 选择fold: {selected_folds}")
    print("="*60)
    
    fold_results = []
    available_folds = []
    
    # 分析每个选定的fold
    for fold_idx in selected_folds:
        if fold_idx < 1:
            print(f"❌ Fold {fold_idx} 编号不能小于1，跳过")
            continue
            
        fold_result = analyze_fold_results(smell_type, fold_idx)
        if fold_result:
            fold_results.append(fold_result)
            available_folds.append(fold_idx)
            print(f"✅ Fold {fold_idx} 分析完成\n")
        else:
            print(f"❌ Fold {fold_idx} 分析失败\n")
    
    if not fold_results:
        print("❌ 没有可用的结果文件")
        return None
    
    print(f"📊 成功分析 {len(fold_results)} 个fold: {available_folds}")
    
    # 计算平均性能
    avg_dnn_metrics = {}
    avg_llm_metrics = {}
    
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc']:
        dnn_values = [result['dnn_original'][metric] for result in fold_results]
        llm_values = [result['llm_enhanced'][metric] for result in fold_results]
        
        avg_dnn_metrics[metric] = {
            'mean': np.mean(dnn_values),
            'std': np.std(dnn_values),
            'min': np.min(dnn_values),
            'max': np.max(dnn_values)
        }
        
        avg_llm_metrics[metric] = {
            'mean': np.mean(llm_values),
            'std': np.std(llm_values),
            'min': np.min(llm_values),
            'max': np.max(llm_values)
        }
    
    # 计算平均改进幅度
    avg_improvements = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc']:
        improve_values = [result['improvements'][metric]['absolute'] for result in fold_results]
        improve_percent_values = [result['improvements'][metric]['percent'] for result in fold_results]
        
        avg_improvements[metric] = {
            'absolute_mean': np.mean(improve_values),
            'absolute_std': np.std(improve_values),
            'percent_mean': np.mean(improve_percent_values),
            'percent_std': np.std(improve_percent_values)
        }
    
    # 输出详细结果
    print("\n" + "📊" + "="*70)
    print("📊 详细性能对比")
    print("="*70)
    
    print(f"\n{'指标':<12} {'DNN原始':<15} {'LLM辅助':<15} {'改进':<15} {'改进率':<15}")
    print("-"*70)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc']:
        dnn_mean = avg_dnn_metrics[metric]['mean']
        llm_mean = avg_llm_metrics[metric]['mean']
        improve_mean = avg_improvements[metric]['absolute_mean']
        improve_percent = avg_improvements[metric]['percent_mean']
        
        print(f"{metric.capitalize():<12} {dnn_mean:<15.4f} {llm_mean:<15.4f} {improve_mean:<+15.4f} {improve_percent:<+15.2f}%")
    
    # 输出标准差信息
    print("\n" + "📊" + "="*70)
    print("📊 标准差信息")
    print("="*70)
    
    print(f"\n{'指标':<12} {'DNN标准差':<15} {'LLM标准差':<15} {'改进标准差':<15}")
    print("-"*70)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc']:
        dnn_std = avg_dnn_metrics[metric]['std']
        llm_std = avg_llm_metrics[metric]['std']
        improve_std = avg_improvements[metric]['absolute_std']
        
        print(f"{metric.capitalize():<12} {dnn_std:<15.4f} {llm_std:<15.4f} {improve_std:<15.4f}")
    
    # 输出范围信息
    print("\n" + "📊" + "="*70)
    print("📊 性能范围")
    print("="*70)
    
    print(f"\n{'指标':<12} {'DNN范围':<20} {'LLM范围':<20}")
    print("-"*70)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc']:
        dnn_min = avg_dnn_metrics[metric]['min']
        dnn_max = avg_dnn_metrics[metric]['max']
        llm_min = avg_llm_metrics[metric]['min']
        llm_max = avg_llm_metrics[metric]['max']
        
        print(f"{metric.capitalize():<12} [{dnn_min:.4f}-{dnn_max:.4f}]  [{llm_min:.4f}-{llm_max:.4f}]")
    
    # 保存综合结果
    summary = {
        'smell_type': smell_type,
        'available_folds': available_folds,
        'fold_results': fold_results,
        'average_dnn_metrics': avg_dnn_metrics,
        'average_llm_metrics': avg_llm_metrics,
        'average_improvements': avg_improvements
    }
    
    output_file = f'performance_analysis_cross_validation_{smell_type}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 详细结果已保存到: {output_file}")
    
    return summary

def main():
    """主函数"""
    # 支持的坏味类型
    smell_types = ['long_method', 'feature_envy', 'blob', 'data_class']
    
    print("🚀 交叉验证结果分析工具")
    print("="*50)
    
    # 选择坏味类型
    print("请选择要分析的坏味类型:")
    for i, smell_type in enumerate(smell_types, 1):
        print(f"{i}. {smell_type}")
    
    try:
        choice = int(input("\n请输入编号 (1-4): "))
        if 1 <= choice <= len(smell_types):
            selected_smell = smell_types[choice - 1]
        else:
            print("❌ 无效选择，默认使用 long_method")
            selected_smell = 'long_method'
    except:
        print("❌ 输入错误，默认使用 long_method")
        selected_smell = 'long_method'
    
    # 直接选择要计算的fold
    print(f"\n📊 请选择要计算的具体fold:")
    print("  输入格式: 单个数字 (如: 1) 或 多个数字用逗号分隔 (如: 1,3,5) 或 范围 (如: 1-3)")
    print("  输入 'all' 或直接回车将计算所有存在的fold")
    
    fold_input = input("请输入fold选择: ").strip()
    
    if fold_input.lower() == 'all' or fold_input == '':
        # 自动检测存在的fold
        selected_folds = []
        for fold_idx in range(1, 6):  # 假设最多5个fold
            file_path = f'results/{selected_smell}_fold{fold_idx}_test_results.json'
            if os.path.exists(file_path):
                selected_folds.append(fold_idx)
        print(f"✅ 选择所有存在的fold: {selected_folds}")
    else:
        selected_folds = []
        # 解析输入
        parts = fold_input.split(',')
        for part in parts:
            part = part.strip()
            if '-' in part:
                # 处理范围
                try:
                    start, end = map(int, part.split('-'))
                    if start < 1:
                        print(f"❌ 范围起始值 {start} 不能小于1")
                    else:
                        selected_folds.extend(range(start, end + 1))
                        print(f"✅ 选择范围: {start}-{end}")
                except:
                    print(f"❌ 无效的范围格式: {part}")
            else:
                # 处理单个数字
                try:
                    fold_num = int(part)
                    if fold_num >= 1:
                        selected_folds.append(fold_num)
                        print(f"✅ 选择fold: {fold_num}")
                    else:
                        print(f"❌ fold编号不能小于1: {fold_num}")
                except:
                    print(f"❌ 无效的fold编号: {part}")
        
        # 去重并排序
        selected_folds = sorted(set(selected_folds))
        
        if not selected_folds:
            print("⚠️  没有有效的fold选择，默认使用所有存在的fold")
            # 自动检测存在的fold
            selected_folds = []
            for fold_idx in range(1, 6):  # 假设最多5个fold
                file_path = f'results/{selected_smell}_fold{fold_idx}_test_results.json'
                if os.path.exists(file_path):
                    selected_folds.append(fold_idx)
        else:
            print(f"✅ 最终选择的fold: {selected_folds}")

    print(f"\n🎯 开始分析 {selected_smell} 的 fold {selected_folds} 结果...")
    
    # 执行分析
    results = analyze_cross_validation(selected_smell, selected_folds)
    
    if results:
        print("\n✅" + "="*60)
        print("✅ 分析完成!")
        print("="*60)
        
        print(f"📊 坏味类型: {results['smell_type']}")
        print(f"📊 可用fold: {results['available_folds']}")
        
        # 输出关键指标
        print("\n📊 关键性能指标 (平均值):")
        for metric in ['accuracy', 'f1_score', 'auc']:
            dnn_mean = results['average_dnn_metrics'][metric]['mean']
            llm_mean = results['average_llm_metrics'][metric]['mean']
            improve = results['average_improvements'][metric]['absolute_mean']
            
            print(f"  {metric.capitalize()}: DNN={dnn_mean:.4f}, LLM={llm_mean:.4f}, 改进={improve:+.4f}")

if __name__ == "__main__":
    main()