#!/usr/bin/env python3
"""
五折交叉验证版本的LLM辅助验证脚本
集成动态LLM介入策略和实时监控
使用固定随机种子666确保可复现性
"""

import os
import sys
import json
import argparse
import numpy as np
import time
import torch
import random
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Set
import threading
import psutil
from datetime import datetime
from sklearn.metrics import f1_score

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from enhanced_detection import EnhancedCodeSmellDetector, load_test_data_for_smell
from config import LLM_CONFIDENCE_THRESHOLD_LOW, LLM_CONFIDENCE_THRESHOLD_HIGH
from llm_validator import LLMValidator

# 小样本学习标杆案例 - 需要在融合权重计算时排除
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

def filter_train_data_for_threshold(train_data: List[Dict], smell_type: str) -> List[Dict]:
    """过滤训练数据，排除标杆案例（用于融合权重计算）"""
    excluded_keys = get_excluded_keys(smell_type)
    if not excluded_keys:
        return train_data
    
    filtered_data = []
    excluded_count = 0
    
    for data in train_data:
        entity_key = data.get('entity_key', '')
        if entity_key in excluded_keys:
            excluded_count += 1
            continue
        filtered_data.append(data)
    
    if excluded_count > 0:
        print(f"🚫 融合权重计算: 排除 {excluded_count} 个标杆案例")
        print(f"📊 剩余训练样本: {len(filtered_data)}")
    
    return filtered_data

class CrossValidationMonitor:
    """五折交叉验证监控器"""
    
    def __init__(self, smell_type, total_folds=5):
        self.smell_type = smell_type
        self.total_folds = total_folds
        self.start_time = time.time()
        self.fold_start_time = {}
        self.fold_progress = {}
        
    def start_fold(self, fold_idx):
        """开始新的fold"""
        self.fold_start_time[fold_idx] = time.time()
        self.fold_progress[fold_idx] = {
            'processed': 0,
            'total': 0,
            'last_update': time.time()
        }
        
    def update_fold_progress(self, fold_idx, processed, total):
        """更新fold进度"""
        if fold_idx not in self.fold_progress:
            self.start_fold(fold_idx)
            
        self.fold_progress[fold_idx]['processed'] = processed
        self.fold_progress[fold_idx]['total'] = total
        self.fold_progress[fold_idx]['last_update'] = time.time()
        
    def monitor_progress(self):
        """监控交叉验证进度"""
        while True:
            try:
                current_time = time.time()
                elapsed_total = current_time - self.start_time
                
                # 计算总体进度
                total_processed = sum(p['processed'] for p in self.fold_progress.values())
                total_samples = sum(p['total'] for p in self.fold_progress.values())
                
                progress_info = []
                for fold_idx in sorted(self.fold_progress.keys()):
                    fold_data = self.fold_progress[fold_idx]
                    if fold_data['total'] > 0:
                        progress = fold_data['processed'] / fold_data['total'] * 100
                        elapsed_fold = current_time - self.fold_start_time.get(fold_idx, current_time)
                        speed = fold_data['processed'] / elapsed_fold if elapsed_fold > 0 else 0
                        progress_info.append(f"Fold{fold_idx}:{progress:.1f}%({speed:.1f}/s)")
                
                if progress_info:
                    print(f"\r[交叉验证] 总体进度:{total_processed}/{total_samples} " + 
                          f"总耗时:{elapsed_total:.0f}s " + " | ".join(progress_info), end="")
                
                time.sleep(5)
                
            except Exception as e:
                print(f"\n监控出错: {e}")
                time.sleep(10)

class CrossValidationLLMStrategy:
    """五折交叉验证LLM策略管理器"""
    
    def __init__(self, model_path, smell_type, fold_idx, use_dynamic_strategy=False, offline=False):
        self.model_path = model_path
        self.smell_type = smell_type
        self.fold_idx = fold_idx
        self.use_dynamic_strategy = use_dynamic_strategy
        self.offline = offline
        
        # 初始化模型和验证器
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        
        if not offline:
            # 使用动态阈值配置初始化LLM验证器
            self.llm_validator = LLMValidator()
            # 为LLM验证器设置动态阈值
            self._setup_dynamic_thresholds()
        
        # 五折交叉验证阈值管理
        self.thresholds_dir = "thresholds"
        os.makedirs(self.thresholds_dir, exist_ok=True)
        
        # 加载或计算阈值
        self.optimal_thresholds = self._load_or_calculate_thresholds()

        # 使用固定阈值0.5
        self.dnn_best_threshold = 0.5

        # 测试阶段不使用动态校准
        self.testing_phase = True
        
        # Token节省策略：跨fold复用LLM调用结果
        self.llm_cache_dir = "llm_cache"
        os.makedirs(self.llm_cache_dir, exist_ok=True)
        self.llm_cache = self._load_llm_cache()
        
    def _get_threshold_file_path(self):
        """获取阈值文件路径"""
        return os.path.join(self.thresholds_dir, f"{self.smell_type}_fold{self.fold_idx}_thresholds.json")
    
    def _setup_dynamic_thresholds(self):
        """为LLM验证器设置动态阈值"""
        # 训练阶段使用标准配置
        self.llm_validator.LLM_CONFIDENCE_THRESHOLD_LOW = LLM_CONFIDENCE_THRESHOLD_LOW
        self.llm_validator.LLM_CONFIDENCE_THRESHOLD_HIGH = LLM_CONFIDENCE_THRESHOLD_HIGH
        print(f"✅ 设置Fold {self.fold_idx}的训练阈值: [{LLM_CONFIDENCE_THRESHOLD_LOW}, {LLM_CONFIDENCE_THRESHOLD_HIGH}]")
    
    def _load_or_calculate_thresholds(self):
        """加载或计算最优阈值"""
        threshold_file = self._get_threshold_file_path()
        
        # 如果阈值文件存在，直接加载
        if os.path.exists(threshold_file):
            try:
                with open(threshold_file, 'r', encoding='utf-8') as f:
                    thresholds = json.load(f)
                print(f"✅ 加载Fold {self.fold_idx}的阈值: {thresholds}")
                return thresholds
            except Exception as e:
                print(f"❌ 加载阈值文件失败: {e}")
        
        # 如果阈值文件不存在，计算最优阈值
        print(f"📊 计算Fold {self.fold_idx}的最优阈值...")
        
        # 加载训练数据
        from enhanced_detection import load_test_data_for_smell
        train_data, all_data_dict = load_test_data_for_smell(
            self.smell_type, self.model_path, self.fold_idx, "training"
        )
        
        if not train_data:
            print("❌ 无法加载训练数据，使用默认阈值")
            return {"low": 0.3, "high": 0.7}
        
        # 在训练集上计算最优阈值
        optimal_thresholds = self._calculate_optimal_thresholds(train_data)
        
        # 保存阈值
        try:
            with open(threshold_file, 'w', encoding='utf-8') as f:
                json.dump(optimal_thresholds, f, indent=2, ensure_ascii=False)
            print(f"✅ 保存Fold {self.fold_idx}的阈值: {optimal_thresholds}")
        except Exception as e:
            print(f"❌ 保存阈值文件失败: {e}")
        
        return optimal_thresholds
    
    def _get_llm_cache_file_path(self):
        """获取LLM缓存文件路径"""
        return os.path.join(self.llm_cache_dir, f"{self.smell_type}_llm_cache.json")
    
    def _load_llm_cache(self):
        """加载LLM调用缓存"""
        cache_file = self._get_llm_cache_file_path()
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                print(f"✅ 加载LLM缓存: {len(cache)} 条记录")
                return cache
            except Exception as e:
                print(f"❌ 加载LLM缓存失败: {e}")
        
        return {}
    
    def _save_llm_cache(self):
        """保存LLM调用缓存（合并模式，不覆盖已有缓存）"""
        cache_file = self._get_llm_cache_file_path()
        
        try:
            # 先加载现有的缓存内容
            existing_cache = {}
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    existing_cache = json.load(f)
            
            # 合并缓存：当前缓存覆盖现有缓存（避免重复）
            merged_cache = {**existing_cache, **self.llm_cache}
            
            # 保存合并后的缓存
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(merged_cache, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 保存LLM缓存: 新增{len(self.llm_cache)}条，合并后共{len(merged_cache)}条记录")
        except Exception as e:
            print(f"❌ 保存LLM缓存失败: {e}")
    
    def _calculate_optimal_thresholds(self, train_data):
        """在训练集上使用15个等距区间方法计算最优置信度阈值"""
        # 排除标杆案例，避免融合权重计算偏差
        train_data = filter_train_data_for_threshold(train_data, self.smell_type)
        
        confidences = []
        labels = []

        # 收集训练集上的置信度和标签
        for data in train_data:
            try:
                # 获取DNN预测
                dnn_result = self.predict_single(data)
                confidences.append(dnn_result['confidence'])
                labels.append(data['label'])
            except Exception as e:
                print(f"❌ 处理训练样本失败: {e}")
                continue

        if not confidences:
            print("❌ 无法获取训练集置信度，使用默认阈值")
            return {"low": 0.3, "high": 0.7}

        # 使用15个等距区间方法（参照run_binary_detection.py）
        import numpy as np
        from sklearn.metrics import f1_score

        print(f"📊 Fold {self.fold_idx}使用15个等距区间方法计算阈值...")

        # 提取所有置信度值
        all_confidences = np.array(confidences)

        # 使用等距区间划分，基于数据范围，划分15个区间
        min_conf = float(np.min(all_confidences))
        max_conf = float(np.max(all_confidences))

        print(f"置信度统计: 最小值={min_conf:.4f}, 最大值={max_conf:.4f}, 样本数={len(all_confidences)}")

        # 步骤1: 先计算全局最优阈值（基于F1分数）
        print(f"\n🔍 步骤1: 计算全局最优阈值...")
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.5
        best_f1 = 0.0

        for threshold in thresholds:
            y_pred = [1 if c >= threshold else 0 for c in confidences]
            f1 = f1_score(labels, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        print(f"✅ 全局最优阈值: {best_threshold:.2f} (F1={best_f1:.4f})")

        # 计算等距步长
        n_intervals = 15
        step = (max_conf - min_conf) / n_intervals

        # 构建等距区间边界
        boundaries = [min_conf + i * step for i in range(n_intervals + 1)]
        boundaries = [round(b, 4) for b in boundaries]  # 保留4位小数提高精度
        boundaries[-1] = max_conf  # 确保包含最大值

        # 构建15个等距区间
        zones = [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]

        # 统计每个区间的表现
        low_confidence_zones = []
        high_confidence_zones = []

        print(f"\n📈 步骤2: 15个等距区间分析 ({min_conf:.4f} - {max_conf:.4f}):")
        print(f"   使用最优阈值 {best_threshold:.2f} 计算准确率\n")

        for i, (start, end) in enumerate(zones):
            # 获取该区间的样本
            zone_samples = [(c, l) for c, l in zip(confidences, labels)
                           if start <= c < end]

            if zone_samples:
                zone_confidences, zone_labels = zip(*zone_samples)
                total = len(zone_samples)
                # 步骤2: 使用最优阈值计算DNN预测准确率
                correct = sum(1 for c, l in zone_samples
                            if (1 if c >= best_threshold else 0) == l)  # 使用最优阈值
                accuracy = correct / total if total > 0 else 0.0

                print(f"区间 {i+1:2d}: [{start:.4f}, {end:.4f}) - 样本数: {total:3d}, 准确率: {accuracy:.2%}")

                # 识别低置信区间（需要LLM介入）
                if accuracy < 0.7 and total >= 5:
                    low_confidence_zones.append((start, end))
                    print(f"  ⚠️  低置信区间（需要LLM介入）")
                # 识别高置信区间（可信任DNN）
                elif accuracy >= 0.85 and total >= 5:
                    high_confidence_zones.append((start, end))
                    print(f"  ✅ 高置信区间（可信任DNN）")
            else:
                print(f"区间 {i+1:2d}: [{start:.4f}, {end:.4f}) - 无样本")
        
        # 合并相邻的低置信区间
        if low_confidence_zones:
            merged_low_zones = self._merge_adjacent_zones(low_confidence_zones)
            
            # 如果存在多个不连续的低置信区间，取最宽的范围覆盖所有低置信区间
            if merged_low_zones:
                if len(merged_low_zones) > 1:
                    # 存在多个不连续的低置信区间，取整体范围
                    best_low = min(zone[0] for zone in merged_low_zones)
                    best_high = max(zone[1] for zone in merged_low_zones)
                    print(f"\n🎯 识别出多个低置信区间，合并为整体范围: [{best_low:.4f}, {best_high:.4f})")
                    print(f"   包含的具体区间: {merged_low_zones}")
                else:
                    # 单个连续低置信区间
                    best_low = merged_low_zones[0][0]
                    best_high = merged_low_zones[0][1]
                    print(f"\n🎯 识别出低置信区间: [{best_low:.4f}, {best_high:.4f})")
            else:
                best_low = 0.3
                best_high = 0.7
                print("⚠️  未识别出低置信区间，使用默认阈值")
        else:
            best_low = 0.3
            best_high = 0.7
            print("⚠️  未识别出低置信区间，使用默认阈值")
        
        print(f"📊 Fold {self.fold_idx}最优阈值: [{best_low:.4f}, {best_high:.4f}]")
        return {"low": best_low, "high": best_high}
    
    def _merge_adjacent_zones(self, zones):
        """合并相邻的置信度区间"""
        if not zones:
            return []
            
        # 按起始值排序
        sorted_zones = sorted(zones, key=lambda x: x[0])
        merged = [list(sorted_zones[0])]
        
        for current in sorted_zones[1:]:
            last = merged[-1]
            # 如果当前区间与上一个区间相邻或重叠，合并它们
            if current[0] <= last[1] + 0.01:  # 允许小的间隔
                last[1] = max(last[1], current[1])
            else:
                merged.append(list(current))
                
        return [tuple(zone) for zone in merged]
    
    def _evaluate_thresholds(self, confidences, labels, low_threshold, high_threshold):
        """评估阈值设置的效果"""
        correct_predictions = 0
        total_predictions = 0
        
        for conf, true_label in zip(confidences, labels):
            # 模拟预测逻辑
            if conf < low_threshold or conf >= high_threshold:
                # DNN直接决策
                dnn_pred = 1 if conf >= 0.5 else 0
                if dnn_pred == true_label:
                    correct_predictions += 1
            else:
                # LLM介入区间，假设LLM能提高准确率
                # 这里简化处理，实际应该使用LLM验证
                correct_predictions += 1  # 假设LLM介入总是正确
            
            total_predictions += 1
        
        return correct_predictions / total_predictions if total_predictions > 0 else 0

    def _get_fixed_threshold(self):
        """获取固定阈值（统一使用0.5）"""
        return 0.5

    def load_model(self):
        """加载模型"""
        try:
            # 检查是否是完整模型还是状态字典
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 根据坏味类型自动确定特征维度
            if "feature_envy" in self.smell_type.lower() or "long_method" in self.smell_type.lower():
                metric_size = 62  # 函数级坏味
            elif "blob" in self.smell_type.lower() or "data_class" in self.smell_type.lower():
                metric_size = 74  # 类级坏味
            else:
                metric_size = 62  # 默认
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # 如果是检查点格式，提取模型结构
                from deepLearningModuleX5 import MyDNN_fusion
                model = MyDNN_fusion(metricSize=metric_size, hidden=128, num_classes=2)
                model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict) and not hasattr(checkpoint, 'eval'):
                # 如果是状态字典格式
                from deepLearningModuleX5 import MyDNN_fusion
                model = MyDNN_fusion(metricSize=metric_size, hidden=128, num_classes=2)
                model.load_state_dict(checkpoint)
            else:
                # 如果是完整模型
                model = checkpoint
                
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"加载模型失败: {e}")
            return None
    
    def predict_single(self, data):
        """对单个样本进行DNN预测"""
        try:
            metrics = data['metrics']
            code_embedding = data['code_embedding']
            
            # 转换为张量
            metrics_tensor = torch.FloatTensor([metrics]).to(self.device)
            code_tensor = torch.FloatTensor([code_embedding]).to(self.device)
            
            # DNN预测
            with torch.no_grad():
                output = self.model(metrics_tensor, code_tensor)
                probabilities = torch.softmax(output, dim=1)
                
                # 获取预测结果和置信度
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities.max().item()
                
                # 对于二分类，如果是类别1，返回对应的概率
                if probabilities.shape[1] == 2:
                    # 返回类别1的概率作为坏味存在的可能性
                    confidence = probabilities[0][1].item()
                    # 使用argmax（等效于0.5阈值）
                    prediction = torch.argmax(probabilities, dim=1).item()
                
            return {
                'prediction': prediction,
                'confidence': confidence
            }
            
        except Exception as e:
            print(f"DNN预测失败: {str(e)}")
            # 返回默认预测结果
            return {
                'prediction': 0,
                'confidence': 0.5
            }
    
    def should_intervene(self, confidence: float, dnn_pred: int) -> Tuple[bool, str]:
        """基于训练集计算的最优阈值决定是否介入LLM验证"""
        
        # 处理不同类型的optimal_thresholds数据结构
        if isinstance(self.optimal_thresholds, dict):
            # 旧格式：字典格式
            low_threshold = self.optimal_thresholds.get("low", 0.3)
            high_threshold = self.optimal_thresholds.get("high", 0.7)
        elif isinstance(self.optimal_thresholds, list) and len(self.optimal_thresholds) > 0:
            # 新格式：风险区间列表
            # 使用第一个风险区间的下限和最后一个风险区间的上限
            first_zone = self.optimal_thresholds[0]
            last_zone = self.optimal_thresholds[-1]
            
            # 解析区间字符串，如"0.5142-0.5449"
            try:
                low_threshold = float(first_zone.get("zone_id", "0.3-0.7").split("-")[0])
                high_threshold = float(last_zone.get("zone_id", "0.3-0.7").split("-")[1])
            except:
                low_threshold = 0.3
                high_threshold = 0.7
        else:
            # 默认值
            low_threshold = 0.3
            high_threshold = 0.7
        
        # 使用训练集计算的最优阈值
        if low_threshold <= confidence < high_threshold:
            return True, f"五折交叉验证：训练集最优区间[{low_threshold:.2f}, {high_threshold:.2f})"
        
        return False, f"五折交叉验证：置信度在可接受范围内（<{low_threshold:.2f}或≥{high_threshold:.2f})"
    
    def add_observation(self, confidence: float, dnn_pred: int, llm_pred: Optional[int], 
                       true_label: Optional[int], was_intervened: bool):
        """在测试阶段仅记录观察数据，不进行实时校准"""
        # 测试阶段不更新阈值，仅用于统计
        pass
    
    def get_strategy_summary(self):
        """获取策略摘要"""
        return {
            "strategy_type": "五折交叉验证",
            "fold_idx": self.fold_idx,
            "optimal_thresholds": self.optimal_thresholds,
            "testing_phase": self.testing_phase
        }

# 删除重复的get_fold_data函数，直接使用enhanced_detection.py中的load_test_data_for_smell函数

def run_single_fold(smell_type, fold_idx, model_path, use_dynamic_strategy=False, 
                   offline=False, show_examples=3, output_dir='results'):
    """运行单个fold的检测（包含训练集配置生成和测试集检测）"""
    print(f"\n{'='*60}")
    print(f"🚀 开始Fold {fold_idx}检测")
    print(f"📊 坏味类型: {smell_type}")
    print(f"🎯 动态策略: {'启用' if use_dynamic_strategy else '传统框架'}")
    print(f"📁 模型路径: {model_path}")
    print(f"{'='*60}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 第一步：运行训练集检测以生成配置
    print(f"📊 Fold {fold_idx} - 运行训练集检测生成配置...")
    
    # 初始化五折交叉验证策略（用于训练集）
    strategy = CrossValidationLLMStrategy(
        model_path=model_path,
        smell_type=smell_type,
        fold_idx=fold_idx,
        use_dynamic_strategy=use_dynamic_strategy,
        offline=offline
    )
    
    # 打印策略摘要
    strategy_summary = strategy.get_strategy_summary()
    print(f"📊 Fold {fold_idx}: 使用策略 - {strategy_summary}")
    
    # 加载训练数据
    train_data, all_data_dict = load_test_data_for_smell(smell_type, model_path, fold_idx, "training")
    
    if not train_data:
        print(f"❌ Fold {fold_idx}无法加载训练数据")
        return {}
    
    print(f"📊 Fold {fold_idx}训练数据: {len(train_data)}个样本")
    
    # 初始化检测器，传入DNN最优分类阈值
    detector = EnhancedCodeSmellDetector(
        model_path=model_path,
        fold_idx=fold_idx,
        smell_type=smell_type,
        all_data_dict=all_data_dict,
        threshold=strategy.dnn_best_threshold
    )
    
    # 执行训练集批量预测（训练集模式）- 添加断点保存功能
    print(f"📊 开始训练集批量预测...")
    
    # 训练集断点文件
    train_checkpoint_file = os.path.join(output_dir, f'{smell_type}_fold{fold_idx}_train_checkpoint.json')
    train_results = []
    
    # 检查断点
    if os.path.exists(train_checkpoint_file):
        with open(train_checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
            train_results = checkpoint.get('results', [])
        print(f"🔄 加载训练集断点，已处理 {len(train_results)} 个样本")
    
    # 获取已处理的实体
    processed_entities = {r['entity_key'] for r in train_results}
    
    # 处理新数据
    new_train_data = [d for d in train_data if d['entity_key'] not in processed_entities]
    
    print(f"🔄 处理新训练样本: {len(new_train_data)}个")
    
    # 分批处理训练集数据
    for i, data in enumerate(new_train_data):
        # 获取DNN预测
        dnn_result = detector.predict_single(data)
        dnn_confidence = dnn_result['confidence']
        dnn_prediction = dnn_result['prediction']
        
        # 使用策略决定是否调用LLM（训练集模式）
        should_use_llm, decision_reason = strategy.should_intervene(dnn_confidence, dnn_prediction)
        
        final_prediction = dnn_prediction
        final_confidence = dnn_confidence
        llm_judgment = None
        llm_reason = None
        
        if should_use_llm and not strategy.offline:
            # 调用LLM验证
            llm_result = detector.predict_with_llm_enhancement(
                metrics=data['metrics'],
                code_embedding=data['code_embedding'],
                entity_key=data['entity_key'],
                smell_type=smell_type,
                all_data_dict=all_data_dict
                # 训练集不传递fold_idx参数，确保使用训练集逻辑（LLM调整）而非测试集逻辑（加权平均）
            )
            
            if llm_result:
                final_prediction = llm_result['final_prediction']
                final_confidence = llm_result['final_confidence']
                llm_judgment = llm_result.get('llm_judgment', '')
                llm_reason = llm_result.get('llm_reason', '')
                llm_suggested_confidence = llm_result.get('llm_suggested_confidence')
        
        # 创建训练集结果记录
        result = {
            'entity_key': data['entity_key'],
            'smell_type': smell_type,
            'dnn_confidence': dnn_confidence,
            'dnn_prediction': dnn_prediction,
            'final_confidence': final_confidence,
            'final_prediction': final_prediction,
            'decision_strategy': decision_reason,
            'was_corrected': should_use_llm and final_prediction != dnn_prediction,
            'true_label': data.get('label', 0),
            'llm_enhanced': should_use_llm
        }
        
        # 当LLM介入时添加详细信息
        if should_use_llm and not strategy.offline:
            if llm_judgment is not None:
                result['llm_judgment'] = llm_judgment
                result['llm_reason'] = llm_reason
                # 添加llm_suggested_confidence字段
                if llm_suggested_confidence is not None:
                    result['llm_suggested_confidence'] = llm_suggested_confidence
            
            if llm_judgment is None:
                result['llm_enhanced_note'] = "代码片段获取失败，跳过LLM验证"
        
        train_results.append(result)
        
        # 定期保存断点（每50个样本）
        if (i + 1) % 50 == 0:
            checkpoint_data = {
                'results': train_results,
                'strategy_summary': strategy_summary,
                'timestamp': datetime.now().isoformat(),
                'processed_count': len(train_results),
                'total_count': len(train_data)
            }
            
            with open(train_checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 已处理 {i+1}/{len(new_train_data)} 训练样本，保存断点...")
    
    print(f"✅ Fold {fold_idx} 训练集批量预测完成，处理样本数: {len(train_results)}")
    
    # 保存最终训练集结果
    train_result_file = os.path.join(output_dir, f'{smell_type}_fold{fold_idx}_train_results.json')
    train_result_data = {
        'smell_type': smell_type,
        'fold': fold_idx,
        'total_samples': len(train_data),
        'processed_samples': len(train_results),
        'results': train_results,
        'timestamp': datetime.now().isoformat(),
        'model_path': model_path,
        'use_dynamic_strategy': use_dynamic_strategy,
        'strategy_summary': strategy_summary,
        'threshold': 0.5  # 固定阈值
    }
    
    with open(train_result_file, 'w', encoding='utf-8') as f:
        json.dump(train_result_data, f, indent=2, ensure_ascii=False)
    
    # 删除断点文件
    if os.path.exists(train_checkpoint_file):
        os.remove(train_checkpoint_file)
    
    print(f"✅ Fold {fold_idx}训练集结果已保存: {train_result_file}")
    
    # 保存训练集处理过程中生成的LLM缓存
    if hasattr(strategy, '_save_llm_cache'):
        strategy._save_llm_cache()
        print(f"💾 Fold {fold_idx}训练集LLM缓存已保存")
    
    # 第二步：基于训练集结果生成配置
    print(f"📊 Fold {fold_idx} - 生成测试集配置...")
    
    # 导入配置生成器
    from generate_config import ConfigGenerator
    
    # 生成配置
    config_file = f"{smell_type}_fold{fold_idx}_config.json"
    config_generator = ConfigGenerator(train_result_file)
    config_success = config_generator.run(config_file)
    
    if config_success:
        print(f"✅ Fold {fold_idx}配置生成成功: {config_file}")
        
        # 加载生成的配置
        config_path = os.path.join(output_dir, config_file)
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                fold_config = json.load(f)
            print(f"✅ Fold {fold_idx}配置加载成功")
            
            # 更新策略的配置
            strategy.optimal_thresholds = fold_config.get('risk_zones', [])
            strategy.fusion_weights = fold_config.get('fusion_weights', {})
            strategy.confidence_mapping = fold_config.get('confidence_mapping', {})
            
            print(f"📊 Fold {fold_idx}策略已更新: {len(strategy.optimal_thresholds)}个风险区间")
            
        except Exception as e:
            print(f"❌ Fold {fold_idx}配置加载失败: {e}")
    else:
        print(f"❌ Fold {fold_idx}配置生成失败，使用默认配置")
    
    # 第三步：运行测试集检测 - 添加断点保存功能
    print(f"📊 Fold {fold_idx} - 运行测试集检测...")
    
    # 加载测试数据
    test_data, _ = load_test_data_for_smell(smell_type, model_path, fold_idx, "testing")
    
    if not test_data:
        print(f"❌ Fold {fold_idx}无法加载测试数据")
        return {}
    
    print(f"测试集大小: {len(test_data)}")
    
    # 测试集断点文件
    test_checkpoint_file = os.path.join(output_dir, f'{smell_type}_fold{fold_idx}_test_checkpoint.json')
    test_results = []
    
    # 检查断点
    if os.path.exists(test_checkpoint_file):
        with open(test_checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
            test_results = checkpoint.get('results', [])
        print(f"🔄 加载测试集断点，已处理 {len(test_results)} 个样本")
    
    # 获取已处理的实体
    processed_entities = {r['entity_key'] for r in test_results}
    
    # 处理新数据
    new_test_data = [d for d in test_data if d['entity_key'] not in processed_entities]
    
    print(f"🔄 处理新测试样本: {len(new_test_data)}个")
    
    # 分批处理测试集数据
    for i, data in enumerate(new_test_data):
        # 获取DNN预测
        dnn_result = detector.predict_single(data)
        dnn_confidence = dnn_result['confidence']
        dnn_prediction = dnn_result['prediction']
        
        # 使用策略决定是否调用LLM（测试集模式）
        should_use_llm, decision_reason = strategy.should_intervene(dnn_confidence, dnn_prediction)
        
        final_prediction = dnn_prediction
        final_confidence = dnn_confidence
        llm_judgment = None
        llm_reason = None
        
        if should_use_llm and not strategy.offline:
            # 调用LLM验证
            llm_result = detector.predict_with_llm_enhancement(
                metrics=data['metrics'],
                code_embedding=data['code_embedding'],
                entity_key=data['entity_key'],
                smell_type=smell_type,
                all_data_dict=all_data_dict,
                fold_idx=fold_idx  # 测试集传递fold_idx参数，确保使用加权平均计算
            )
            
            if llm_result:
                final_prediction = llm_result['final_prediction']
                final_confidence = llm_result['final_confidence']
                llm_judgment = llm_result.get('llm_judgment', '')
                llm_reason = llm_result.get('llm_reason', '')
        
        # 创建测试集结果记录
        result = {
            'entity_key': data['entity_key'],
            'smell_type': smell_type,
            'dnn_confidence': dnn_confidence,
            'dnn_prediction': dnn_prediction,
            'final_confidence': final_confidence,
            'final_prediction': final_prediction,
            'decision_strategy': decision_reason,
            'was_corrected': should_use_llm and final_prediction != dnn_prediction,
            'true_label': data.get('label', 0),
            'llm_enhanced': should_use_llm
        }
        
        # 当LLM介入时添加详细信息
        if should_use_llm and not strategy.offline:
            if llm_judgment is not None:
                result['llm_judgment'] = llm_judgment
                result['llm_reason'] = llm_reason
            
            if llm_judgment is None:
                result['llm_enhanced_note'] = "代码片段获取失败，跳过LLM验证"
            
            # 添加llm_suggested_confidence字段（如果有）
            if llm_result and 'llm_suggested_confidence' in llm_result:
                result['llm_suggested_confidence'] = llm_result['llm_suggested_confidence']
        
        test_results.append(result)
        
        # 定期保存断点（每50个样本）
        if (i + 1) % 50 == 0:
            checkpoint_data = {
                'results': test_results,
                'strategy_summary': strategy_summary,
                'timestamp': datetime.now().isoformat(),
                'processed_count': len(test_results),
                'total_count': len(test_data)
            }
            
            with open(test_checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 已处理 {i+1}/{len(new_test_data)} 测试样本，保存断点...")
    
    print(f"✅ Fold {fold_idx} 测试集批量预测完成，处理样本数: {len(test_results)}")

    # 使用固定阈值0.5计算最终预测
    print(f"\n📊 使用固定阈值0.5计算最终预测...")
    fusion_best_threshold = 0.5

    # 对所有样本使用0.5阈值计算final_prediction
    llm_enhanced_count = 0
    llm_not_enhanced_count = 0

    for result in test_results:
        if result.get('llm_enhanced', False):
            # LLM介入的样本：使用0.5阈值
            result['final_prediction'] = 1 if result['final_confidence'] >= 0.5 else 0
            llm_enhanced_count += 1
        else:
            # LLM未介入的样本：保持DNN原始预测
            result['final_prediction'] = result['dnn_prediction']
            llm_not_enhanced_count += 1

        # 更新was_corrected字段
        result['was_corrected'] = result['final_prediction'] != result['dnn_prediction']

    print(f"✅ 已更新预测结果：")
    print(f"   - LLM介入样本 ({llm_enhanced_count}个)：使用阈值 0.5")
    print(f"   - LLM未介入样本 ({llm_not_enhanced_count}个)：保持DNN原始预测")

    # 保存最终测试集结果
    test_result_file = os.path.join(output_dir, f'{smell_type}_fold{fold_idx}_test_results.json')
    test_result_data = {
        'smell_type': smell_type,
        'fold': fold_idx,
        'total_samples': len(test_data),
        'processed_samples': len(test_results),
        'results': test_results,
        'timestamp': datetime.now().isoformat(),
        'model_path': model_path,
        'use_dynamic_strategy': use_dynamic_strategy,
        'strategy_summary': strategy_summary,
        'threshold': 0.5  # 固定阈值
    }

    with open(test_result_file, 'w', encoding='utf-8') as f:
        json.dump(test_result_data, f, indent=2, ensure_ascii=False)
    
    # 删除断点文件
    if os.path.exists(test_checkpoint_file):
        os.remove(test_checkpoint_file)
    
    print(f"\n✅ Fold {fold_idx} 完成，测试集结果保存至: {test_result_file}")
    
    # 保存LLM缓存
    if hasattr(strategy, '_save_llm_cache'):
        strategy._save_llm_cache()
    
    return test_result_data



def main():
    """主函数"""
    # 设置全局固定随机种子666确保可复现性
    random.seed(666)
    np.random.seed(666)
    torch.manual_seed(666)
    
    parser = argparse.ArgumentParser(description='五折交叉验证版本的LLM辅助验证脚本')
    parser.add_argument('--smell-type', required=True,
                       choices=['feature_envy', 'long_method', 'blob', 'data_class'],
                       help='检测的坏味类型')
    parser.add_argument('--model-base-path', required=True,
                       help='模型基础路径，会自动查找fold模型')
    parser.add_argument('--dynamic-strategy', action='store_true',
                       help='启用动态LLM介入策略')
    parser.add_argument('--offline', action='store_true',
                       help='离线模式，跳过LLM验证')
    parser.add_argument('--show-examples', type=int, default=3,
                       help='显示示例数量')
    parser.add_argument('--total-folds', type=int, default=5,
                       help='总fold数，默认为5')
    parser.add_argument('--all-folds', action='store_true',
                       help='运行所有fold（等同于--total-folds 5）')
    parser.add_argument('--output-dir', default='results',
                       help='输出目录')

    
    args = parser.parse_args()
    
    # 处理--all-folds参数
    if args.all_folds:
        args.total_folds = 5
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🚀 启动五折交叉验证LLM辅助验证脚本")
    print("🎯 使用固定随机种子666确保可复现性")
    print(f"📊 坏味类型: {args.smell_type}")
    print(f"🎯 动态策略: {'启用' if args.dynamic_strategy else '传统框架'}")
    print(f"📊 总fold数: {args.total_folds}")
    print(f"📁 输出目录: {args.output_dir}")
    print("=" * 60)
    
    # 创建监控器
    monitor = CrossValidationMonitor(args.smell_type, args.total_folds)
    
    # 启动监控线程
    progress_thread = threading.Thread(target=monitor.monitor_progress, daemon=True)
    progress_thread.start()
    
    print("📊 监控线程已启动...")
    
    # 运行所有fold
    all_results = []
    
    for fold_idx in range(1, args.total_folds + 1):
        # 构建模型路径
        model_path = args.model_base_path.replace('_fold1_', f'_fold{fold_idx}_')
        
        if not os.path.exists(model_path):
            print(f"❌ Fold {fold_idx}: 模型文件不存在: {model_path}")
            continue
        
        # 开始fold
        monitor.start_fold(fold_idx)
        
        # 运行单个fold
        try:
            result = run_single_fold(
                smell_type=args.smell_type,
                fold_idx=fold_idx,
                model_path=model_path,
                use_dynamic_strategy=args.dynamic_strategy,
                offline=args.offline,
                show_examples=args.show_examples,
                output_dir=args.output_dir
            )
            all_results.append(result)
            
        except Exception as e:
            print(f"❌ Fold {fold_idx} 运行失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ 五折交叉验证完成!")
    print("📊 请使用独立的 analyze_cross_validation.py 脚本进行结果分析")

if __name__ == "__main__":
    # 使用示例
    if len(sys.argv) == 1:
        print("=" * 60)
        print("🚀 五折交叉验证LLM辅助验证脚本")
        print("=" * 60)
        print("\n📋 使用示例:")
        print("python run_binary_detection_cross_validation.py --smell-type feature_envy --model-base-path model/MyDNN_fusion_feature_envy_fold1_binary.pth")
        print("python run_binary_detection_cross_validation.py --smell-type blob --model-base-path model/MyDNN_fusion_blob_fold1_binary.pth --dynamic-strategy")
        print("python run_binary_detection_cross_validation.py --smell-type long_method --model-base-path model/MyDNN_fusion_long_method_fold1_binary.pth --offline")
        print("python run_binary_detection_cross_validation.py --smell-type data_class --model-base-path model/MyDNN_fusion_data_class_fold1_binary.pth")
    else:
        main()