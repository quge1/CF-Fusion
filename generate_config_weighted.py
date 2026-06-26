#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件生成器
基于训练集binary_long_method_results_reality_based.json生成测试集配置
"""

import json
import numpy as np
from collections import defaultdict
import os
from datetime import datetime
from sklearn.metrics import f1_score

class ConfigGenerator:
    def __init__(self, training_data_path):
        """初始化配置生成器（基于单个fold的训练集结果）"""
        self.training_data_path = training_data_path
        self.training_data = None
        self.risk_zones = []
        self.confidence_mapping = {}
        self.fusion_weights = {}
        self.fold_idx = None  # 存储fold索引
        
    def load_training_data(self):
        """加载单个fold的训练数据"""
        try:
            with open(self.training_data_path, 'r', encoding='utf-8') as f:
                self.training_data = json.load(f)
            
            # 提取fold索引
            self.fold_idx = self.training_data.get('fold', 0)
            results = self.training_data.get('results', [])
            
            print(f"成功加载Fold {self.fold_idx}训练数据，共{len(results)}条记录")
        except Exception as e:
            print(f"加载训练数据失败: {e}")
            return False
        return True
    
    def calculate_risk_zones(self):
        """计算风险区间（使用DNN准确率判断）"""
        zones = []
        
        # 基于实际数据计算最大最小置信度
        confidences = [item.get('dnn_confidence', 0) for item in self.training_data.get('results', []) if 'dnn_confidence' in item]
        
        if not confidences:
            print("警告：数据中没有置信度信息，使用默认区间")
            return zones
        
        min_conf = min(confidences)
        max_conf = max(confidences)
        
        print(f"置信度统计: 最小值={min_conf:.4f}, 最大值={max_conf:.4f}, 样本数={len(confidences)}")
        
        # 使用等距区间划分，基于实际数据范围，划分15个区间
        n_intervals = 15
        interval_width = (max_conf - min_conf) / n_intervals
        
        for i in range(n_intervals):
            lower = min_conf + i * interval_width
            upper = lower + interval_width
            
            # 获取该区间的样本（基于dnn_confidence）
            zone_samples = [
                item for item in self.training_data.get('results', [])
                if lower <= item.get('dnn_confidence', 0) < upper
            ]
            
            if len(zone_samples) >= 5:  # 最小样本数要求
                # 使用DNN原始预测计算准确率（而不是final_prediction）
                correct = sum(1 for item in zone_samples 
                            if item.get('dnn_prediction', 0) == item.get('true_label', 0))
                accuracy = correct / len(zone_samples)
                
                if accuracy < 0.7:  # 准确率阈值
                    zones.append({
                        'zone_id': f"{lower:.4f}-{upper:.4f}",
                        'accuracy': round(accuracy, 3),
                        'sample_count': len(zone_samples),
                        'requires_llm': True
                    })
        
        self.risk_zones = zones
        print(f"识别出{len(zones)}个风险区间（基于DNN准确率）")
        return zones
    
    def build_confidence_mapping(self):
        """构建置信度映射表（基于LLM参与数据，排除冷启动期）"""
        results = self.training_data.get('results', [])
        
        # 筛选有LLM参与的数据，排除冷启动期（前50个样本）
        llm_samples = [
            item for item in results[50:]  # 排除前50个冷启动样本
            if item.get('llm_suggested_confidence') is not None
        ]
        
        if not llm_samples:
            print("警告：没有找到LLM参与的有效样本")
            return {}
        
        # 按LLM输出值排序并去重
        unique_mappings = {}
        for item in llm_samples:
            llm_val = round(item.get('llm_suggested_confidence'), 3)
            dnn_standard = item.get('final_confidence', llm_val)
            if llm_val not in unique_mappings:
                unique_mappings[llm_val] = dnn_standard
        
        # 排序映射表
        sorted_llm_values = sorted(unique_mappings.keys())
        exact_mapping = {str(k): unique_mappings[k] for k in sorted_llm_values}
        
        # 构建插值区间
        interpolation_ranges = []
        for i in range(len(sorted_llm_values) - 1):
            interpolation_ranges.append({
                'llm_min': sorted_llm_values[i],
                'llm_max': sorted_llm_values[i + 1],
                'dnn_min': unique_mappings[sorted_llm_values[i]],
                'dnn_max': unique_mappings[sorted_llm_values[i + 1]]
            })
        
        self.confidence_mapping = {
            'exact_values': exact_mapping,
            'interpolation_ranges': interpolation_ranges,
            'boundaries': {
                'min_llm': min(sorted_llm_values),
                'max_llm': max(sorted_llm_values),
                'min_dnn': min(exact_mapping.values()),
                'max_dnn': max(exact_mapping.values())
            },
            'sample_count': len(llm_samples),
            'cold_start_excluded': 50
        }
        
        print(f"构建映射表完成，共{len(exact_mapping)}个精确映射（基于{len(llm_samples)}个LLM参与样本）")
        return self.confidence_mapping
    
    def calculate_fusion_weights(self):
        """计算融合权重（基于F1-score的网格搜索）"""
        results = self.training_data.get('results', [])

        # 统一使用 0.5 阈值
        threshold = 0.5
        print(f"使用固定阈值 {threshold:.2f} 计算权重")

        # 筛选有LLM参与的数据，排除冷启动期（前50个样本）
        llm_participated_data = [
            item for item in results[50:]  # 排除前50个冷启动样本
            if item.get('llm_suggested_confidence') is not None
        ]

        if not llm_participated_data:
            print("警告：没有LLM参与的有效数据")
            return {'alpha': 0.5}

        # 提取真实标签和置信度
        true_labels = []
        dnn_confidences = []
        llm_confidences = []

        for item in llm_participated_data:
            true_labels.append(item.get('true_label'))
            dnn_confidences.append(item.get('dnn_confidence', 0.5))
            llm_confidences.append(item.get('llm_suggested_confidence', 0.5))

        # 网格搜索最佳alpha
        best_alpha = 0.5
        best_f1 = 0.0

        # 遍历alpha值（0.0到1.0，步长0.05）
        for alpha in np.arange(0.0, 1.05, 0.05):
            # 计算融合置信度
            fusion_confidences = []
            for dnn_conf, llm_conf in zip(dnn_confidences, llm_confidences):
                fusion_conf = alpha * dnn_conf + (1 - alpha) * llm_conf
                fusion_confidences.append(fusion_conf)

            # 生成预测
            predictions = [1 if conf >= threshold else 0 for conf in fusion_confidences]

            # 计算F1-score
            current_f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)

            # 更新最佳alpha
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_alpha = alpha

        # 计算最终权重
        beta = 1 - best_alpha

        # 计算DNN和LLM的准确率（用于参考）
        dnn_predictions = [1 if conf >= threshold else 0 for conf in dnn_confidences]
        llm_predictions = [1 if conf >= threshold else 0 for conf in llm_confidences]

        dnn_accuracy = sum(1 for dnn_pred, true in zip(dnn_predictions, true_labels) if dnn_pred == true) / len(true_labels)
        llm_accuracy = sum(1 for llm_pred, true in zip(llm_predictions, true_labels) if llm_pred == true) / len(true_labels)

        self.fusion_weights = {
            'alpha': round(best_alpha, 4),  # DNN权重
            'beta': round(beta, 4),  # LLM权重
            'best_f1': round(best_f1, 4),  # 最佳F1-score
            'dnn_accuracy': round(dnn_accuracy, 4),
            'llm_accuracy': round(llm_accuracy, 4),
            'threshold': threshold,  # 固定阈值
            'total_samples': len(llm_participated_data),
            'llm_samples': len(llm_participated_data),
            'cold_start_excluded': 50
        }

        print(f"权重计算完成: alpha(DNN)={best_alpha:.4f}, beta(LLM)={beta:.4f} (基于{len(llm_participated_data)}个LLM参与样本)")
        print(f"  最佳F1-score: {best_f1:.4f}")
        print(f"  DNN准确率: {dnn_accuracy:.4f}, LLM准确率: {llm_accuracy:.4f}")
        return self.fusion_weights
    
    def generate_config(self, output_path):
        """生成基于单个fold训练集的配置文件"""
        # 计算所有zone_id的总体范围
        all_zone_range = None
        if self.risk_zones:
            # 提取所有zone_id的最小值和最大值
            all_lower_bounds = []
            all_upper_bounds = []
            
            for zone in self.risk_zones:
                zone_id = zone['zone_id']
                lower, upper = map(float, zone_id.split('-'))
                all_lower_bounds.append(lower)
                all_upper_bounds.append(upper)
            
            min_lower = min(all_lower_bounds)
            max_upper = max(all_upper_bounds)
            all_zone_range = f"{min_lower:.2f}-{max_upper:.2f}"
        
        # 从训练数据中提取smell_type
        smell_type = self.training_data.get('smell_type', 'unknown')
        
        config = {
            "smell_type": smell_type,
            "fold": self.fold_idx,
            "all_zone_range": all_zone_range,  # 添加所有zone_id的总体范围
            "risk_zones": self.risk_zones,
            "confidence_mapping": self.confidence_mapping,
            "fusion_weights": self.fusion_weights,
            "metadata": {
                "training_samples": len(self.training_data.get('results', [])),
                "llm_outputs": len([r for r in self.training_data.get('results', []) 
                                  if r.get('llm_suggested_confidence') is not None]),
                "exact_mappings": len(self.confidence_mapping.get('exact_values', {})),
                "interpolation_ranges": len(self.confidence_mapping.get('interpolation_ranges', [])),
                "config_generated_at": datetime.now().isoformat(),
                "based_on_fold": self.fold_idx
            }
        }
        
        # 确保results_weighted文件夹存在
        results_dir = 'results_weighted'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # 使用完整路径
        full_path = os.path.join(results_dir, output_path)
        
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"配置文件已生成: {full_path}")
            return True
        except Exception as e:
            print(f"生成配置文件失败: {e}")
            return False
    
    def run(self, output_path="long_method_config.json"):
        """运行完整流程（基于单个fold训练集）"""
        print(f"=== 开始生成Fold {self.fold_idx if self.fold_idx is not None else 'unknown'}配置文件 ===")
        
        if not self.load_training_data():
            return False
        
        print(f"基于Fold {self.fold_idx}训练集生成配置...")
        
        self.calculate_risk_zones()
        self.build_confidence_mapping()
        self.calculate_fusion_weights()
        
        return self.generate_config(output_path)

def main():
    """主函数 - 支持命令行参数，根据坏味类型和fold生成配置"""
    import argparse
    
    parser = argparse.ArgumentParser(description='基于单个fold训练集生成配置')
    parser.add_argument('--smell-type', required=True,
                       choices=['feature_envy', 'long_method', 'blob', 'data_class'],
                       help='坏味类型')
    parser.add_argument('--fold', type=int, required=True,
                       help='fold索引 (1-5)')
    parser.add_argument('--results-dir', default='results_weighted',
                       help='结果目录路径')
    
    args = parser.parse_args()
    
    # 构建训练结果文件路径
    train_result_file = os.path.join(args.results_dir, f"{args.smell_type}_fold{args.fold}_train_results.json")
    
    # 检查文件是否存在
    if not os.path.exists(train_result_file):
        print(f"训练结果文件不存在: {train_result_file}")
        print("请先运行五折交叉验证脚本生成训练集结果")
        return
    
    # 构建输出配置文件名
    output_path = f"{args.smell_type}_fold{args.fold}_config.json"
    
    print(f"开始为{args.smell_type} Fold {args.fold}生成配置...")
    print(f"训练数据: {train_result_file}")
    print(f"输出配置: {output_path}")
    
    # 创建生成器并运行
    generator = ConfigGenerator(train_result_file)
    success = generator.run(output_path)
    
    if success:
        print(f"\n{args.smell_type} Fold {args.fold}配置文件生成成功")
        print(f"输出文件: {output_path}")
        print("此配置将用于对应fold的测试集检测")
    else:
        print(f"\n{args.smell_type} Fold {args.fold}配置文件生成失败")

if __name__ == "__main__":
    main()