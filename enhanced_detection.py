"""
增强版代码坏味检测器
集成DNN + LLM协同架构
基于idea1.txt的设计实现
"""

import torch
import json
import time
import os
import random
from typing import Dict, List, Tuple, Any
from deepLearningModuleX5 import MyDNN_fusion
from llm_validator import llm_validator

class EnhancedCodeSmellDetector:
    """DNN+LLM协同检测器，支持五折交叉验证"""
    
    def __init__(self, model_path: str, metric_size: int = None, fold_idx: int = None, smell_type: str = None, all_data_dict: Dict = None, threshold: float = 0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dnn_model = self._load_dnn_model(model_path, metric_size)
        self.llm_validator = llm_validator
        self.fold_idx = fold_idx
        self.smell_type = smell_type
        self.all_data_dict = all_data_dict or {}
        self.threshold = threshold  # DNN分类阈值
        
        print(f"[OK] DNN模型加载成功: {model_path}")
        if fold_idx is not None:
            print(f"[INFO] 五折交叉验证模式: Fold {fold_idx}")
        if smell_type is not None:
            print(f"[INFO] 坏味类型: {smell_type}")
        
    def _load_dnn_model(self, model_path: str, metric_size: int = None) -> MyDNN_fusion:
        """加载DNN模型"""
        # 从checkpoint文件名判断模型类型
        is_severity = "severity" in str(model_path).lower()
        
        if metric_size is None:
            # 如果没有指定特征维度，根据模型类型自动选择
            if is_severity:
                metric_size = 74  # 严重性模型的默认输入维度
            else:
                # 二分类模型 - 根据坏味类型确定特征维度
                if "feature_envy" in str(model_path).lower() or "long_method" in str(model_path).lower():
                    metric_size = 62
                elif "blob" in str(model_path).lower() or "data_class" in str(model_path).lower():
                    metric_size = 74
                else:
                    metric_size = 62  # 默认
        
        num_classes = 4 if is_severity else 2
        
        model = MyDNN_fusion(
            metricSize=metric_size,
            hidden=128,
            num_classes=num_classes
        ).to(self.device)
        
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model
    
    def _get_code_snippet_for_entity(self, entity_key: str, 
                         all_data_dict: Dict, fold_idx: int = None) -> str:
        """获取代码片段 - 仅使用精确匹配，支持五折交叉验证"""
        try:
            # 获取项目根目录
            project_root = os.path.dirname(os.path.abspath(__file__))
            
            # 统一使用数据集目录，因为我们只有dataset文件夹
            source_code_dir = os.path.join(project_root, "dataset", "sourceCode")
            
            # 构建精确文件名
            filename_pattern = f"{entity_key}.java"
            target_file = os.path.join(source_code_dir, filename_pattern)
            
            # 仅使用精确匹配，不再使用模糊匹配
            if os.path.exists(target_file):
                with open(target_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # 验证文件内容是否包含预期的类或方法
                    if self._validate_file_content(content, entity_key):
                        # 智能截断：保持方法/类结构完整
                        lines = content.split('\n')
                        if len(lines) > 200:
                            # 保留前200行，通常能展示完整结构
                            content = '\n'.join(lines[:200]) + "\n// ... 代码截断，保留核心结构 ..."
                        elif len(content) > 8000:
                            content = content[:8000] + "\n// ... 代码截断 ..."
                        
                        return content
                    else:
                        print(f"⚠️ 文件内容验证失败: {entity_key}")
                        return ""
            else:
                print(f"❌ 文件不存在: {target_file}")
                return ""
                
            return ""
        except Exception as e:
            print(f"加载代码片段失败 {entity_key}: {str(e)}")
            return ""

    def _validate_file_content(self, content: str, entity_key: str) -> bool:
        """验证文件内容是否包含预期的类或方法"""
        try:
            parts = entity_key.split('__')
            if len(parts) < 7:
                return False
            
            entity_type = parts[0]  # class 或 function
            
            if entity_type == "class":
                # 验证类名 - parts[6] 是类名
                class_name = parts[6]
                class_pattern = f"class {class_name}"
                return class_pattern in content
            elif entity_type == "function":
                # 对于函数，验证内容不为空即可，因为方法名需要从代码中提取
                # 实际的方法名验证在后续处理中进行
                return len(content.strip()) > 0
            
            return True  # 默认通过验证
        except Exception:
            return False
    
    def predict_with_llm_enhancement(self, metrics, code_embedding, entity_key, smell_type, all_data_dict, fold_idx: int = None):
        """使用LLM增强的预测，修复问题1和问题2，支持五折交叉验证"""
        # 获取DNN预测
        dnn_result = self.predict_single({'metrics': metrics, 'code_embedding': code_embedding})
        dnn_confidence = dnn_result['confidence']
        dnn_prediction = dnn_result['prediction']
        
        # 检查是否需要LLM验证
        needs_llm_validation = self.should_use_llm_validation(dnn_confidence, dnn_prediction, smell_type)
        
        if not needs_llm_validation:
            # DNN直接决策
            return {
                'final_prediction': dnn_prediction,
                'final_confidence': dnn_confidence,
                'llm_enhanced': False,
                'decision_strategy': f"DNN直接决策：置信度{dnn_confidence:.3f}在可接受范围内"
            }
        
        # 获取代码片段（支持五折交叉验证）
        code_snippet = self._get_code_snippet_for_entity(entity_key, all_data_dict, fold_idx)
        
        if not code_snippet:
            # 代码片段获取失败
            return {
                'final_prediction': dnn_prediction,
                'final_confidence': dnn_confidence,
                'llm_enhanced': [True, "代码片段获取失败，跳过LLM验证"],
                'decision_strategy': f"LLM介入失败：无法获取代码片段"
            }
        
        # 调用LLM验证
        llm_result = self.llm_validator.validate_boundary_case(
            code=code_snippet,
            smell_type=smell_type,
            dnn_confidence=dnn_confidence,
            metrics=metrics,
            severity_mode=False,
            entity_key=entity_key
        )
        
        if not llm_result or not llm_result.get('use_llm', False):
            # LLM验证失败或选择不使用
            return {
                'final_prediction': dnn_prediction,
                'final_confidence': dnn_confidence,
                'llm_enhanced': [True, "LLM验证失败"],
                'decision_strategy': f"LLM介入失败：验证未通过"
            }
        
        # LLM验证成功，返回增强结果
        # 检查LLM验证是否真正成功（不是fallback到原始值的情况）
        confidence_source = llm_result.get('confidence_source', 'llm_returned')
        parsing_warning = llm_result.get('parsing_warning', '')
        
        # 如果LLM验证失败（fallback到原始值或解析警告），则视为验证失败
        if confidence_source == 'fallback_to_original' or '未找到判断字段' in parsing_warning:
            # LLM验证失败，不使用LLM结果
            return {
                'final_prediction': dnn_prediction,
                'final_confidence': dnn_confidence,
                'llm_enhanced': [True, "LLM解析失败"],
                'decision_strategy': f"LLM介入失败：解析失败 - {parsing_warning}",
                'llm_judgment': "验证失败",
                'llm_reason': f"LLM解析失败: {parsing_warning}"
            }
        
        # 根据当前模式选择置信度计算方式
        if fold_idx is not None:  # 测试集模式（五折交叉验证）
            # 测试集：使用加权平均计算final_confidence
            fusion_config = self._load_fusion_config(smell_type, fold_idx)

            # 使用固定阈值0.5
            threshold = 0.5
            
            # 从配置中读取alpha（DNN权重）
            if fusion_config and 'fusion_weights' in fusion_config:
                alpha = fusion_config['fusion_weights'].get('alpha', 0.5)
                beta = fusion_config['fusion_weights'].get('beta', 0.5)
                # 验证权重和是否为1（允许0.001的误差）
                if abs(alpha + beta - 1.0) > 0.001:
                    print(f"警告：权重和不为1，alpha={alpha}, beta={beta}")
                    # 归一化权重
                    total = alpha + beta
                    alpha = alpha / total
                    beta = beta / total
            else:
                # 如果配置不存在，使用默认权重
                alpha = 0.5
                beta = 0.5
                print("警告：未找到融合权重配置，使用默认权重0.5")

            llm_suggested_confidence = llm_result.get('llm_suggested_confidence', dnn_confidence)
            final_confidence = alpha * dnn_confidence + beta * llm_suggested_confidence
            decision_strategy = f"测试集加权平均: {dnn_confidence:.3f}×{alpha:.3f} + {llm_suggested_confidence:.3f}×{beta:.3f}"
        else:
            # 训练集：使用llm_validator.py中的调整逻辑
            # 关键修复：训练集模式下，基于当前dnn_confidence重新计算final_confidence
            # 而不是直接使用缓存中的adjusted_confidence值

            # 获取LLM判断和建议的置信度
            llm_judgment = llm_result.get('llm_judgment', '否')
            llm_suggested_confidence = llm_result.get('llm_suggested_confidence', dnn_confidence)

            # 使用llm_validator.py中的调整公式重新计算
            confidence_change = abs(llm_suggested_confidence - dnn_confidence)

            if llm_judgment == "否":
                # LLM判断为"否"时，置信度应该降低
                final_confidence = max(0.1, dnn_confidence - confidence_change * 0.8)
            elif llm_judgment == "是":
                # LLM判断为"是"时，置信度应该升高
                final_confidence = min(0.9, dnn_confidence + confidence_change * 0.8)
            else:
                # 其他情况使用原始置信度
                final_confidence = dnn_confidence

            decision_strategy = f"训练集LLM调整(重新计算): {dnn_confidence:.3f}→{final_confidence:.3f} (判断:{llm_judgment}, 建议:{llm_suggested_confidence:.3f})"

            # 训练集模式使用固定阈值0.5
            threshold = 0.5

        # 在LLM验证成功的返回结果中添加llm_suggested_confidence字段
        # 使用固定阈值0.5计算final_prediction
        result = {
            'final_prediction': 1 if final_confidence >= 0.5 else 0,
            'final_confidence': final_confidence,
            'llm_enhanced': True,
            'decision_strategy': decision_strategy,
            'llm_judgment': llm_result.get('llm_judgment', ''),
            'llm_reason': llm_result.get('reason', ''),
            'llm_suggested_confidence': llm_result.get('llm_suggested_confidence'),
            'threshold': 0.5  # 固定阈值
        }

        return result

    def batch_predict_enhanced(self, test_data, strategy=None, show_examples: int = 0, is_testing: bool = False):
        """批量预测，支持五折交叉验证策略
        
        Args:
            test_data: 测试数据
            strategy: 策略类
            show_examples: 显示示例数量
            is_testing: 是否为测试集（True=测试集，False=训练集）
        """
        results = []
        
        print(f"处理新样本: {len(test_data)}个 (模式: {'测试集' if is_testing else '训练集'})")
        
        for i, data in enumerate(test_data):
            entity_key = data['entity_key']
            
            # 获取DNN预测
            dnn_result = self.predict_single(data)
            dnn_confidence = dnn_result['confidence']
            dnn_prediction = dnn_result['prediction']
            
            # 决策逻辑：使用策略类或默认逻辑
            if strategy:
                # 使用策略类决定是否介入LLM验证
                needs_llm_validation, decision_reason = strategy.should_intervene(dnn_confidence, dnn_prediction)
            else:
                # 默认逻辑：使用固定区间[0.3, 0.7)
                needs_llm_validation = self.should_use_llm_validation(dnn_confidence, dnn_prediction, self.smell_type)
                decision_reason = f"默认策略：固定区间[0.3, 0.7)"
            
            final_prediction = dnn_prediction
            final_confidence = dnn_confidence
            llm_judgment = None
            llm_reason = None
            was_corrected = False
            debug_info = decision_reason
            
            if needs_llm_validation:
                # 获取代码片段（支持五折交叉验证）
                code_snippet = self._get_code_snippet_for_entity(entity_key, self.all_data_dict, self.fold_idx)
                
                if code_snippet:
                    # Token节省策略：检查跨fold缓存
                    cache_key = f"{entity_key}_{self.smell_type}"
                    
                    # 如果使用策略类且有缓存，优先使用缓存
                    if strategy and hasattr(strategy, 'llm_cache') and cache_key in strategy.llm_cache:
                        llm_result = strategy.llm_cache[cache_key]
                        print(f"✅ 使用缓存结果: {entity_key}")
                    else:
                        # 调用LLM验证
                        llm_result = self.llm_validator.validate_boundary_case(
                            code=code_snippet,
                            smell_type=self.smell_type,
                            dnn_confidence=dnn_confidence,
                            metrics=data['metrics'],
                            severity_mode=False,
                            entity_key=entity_key
                        )
                        
                        # 如果使用策略类，将结果保存到缓存
                        if strategy and hasattr(strategy, 'llm_cache') and llm_result:
                            strategy.llm_cache[cache_key] = llm_result
                            print(f"💾 保存LLM结果到缓存: {entity_key}")
                            
                            # 立即保存缓存到文件，确保训练阶段也能保存（实时保存）
                            if hasattr(strategy, '_save_llm_cache'):
                                strategy._save_llm_cache()
                                print(f"💾 实时保存策略缓存到文件: {entity_key}")
                    
                    if llm_result and llm_result.get('use_llm', False):
                        # 检查LLM验证是否真正成功（不是fallback到原始值的情况）
                        confidence_source = llm_result.get('confidence_source', 'llm_returned')
                        parsing_warning = llm_result.get('parsing_warning', '')
                        
                        # 如果LLM验证失败（fallback到原始值或解析警告），则视为验证失败
                        if confidence_source == 'fallback_to_original' or '未找到判断字段' in parsing_warning:
                            # LLM验证失败，不使用LLM结果
                            final_prediction = dnn_prediction
                            final_confidence = dnn_confidence
                            llm_judgment = "验证失败"
                            llm_reason = f"LLM解析失败: {parsing_warning}"
                            llm_suggested_confidence = None
                            was_corrected = False
                            debug_info = f"LLM验证失败，使用DNN原始结果: {parsing_warning}"
                        else:
                            # LLM验证成功，根据训练集/测试集使用不同的final_confidence计算方式
                            if is_testing:
                                # 测试集：使用加权平均计算final_confidence
                                # 从配置文件动态获取权重（传递fold信息）
                                fusion_config = self._load_fusion_config(self.smell_type, self.fold_idx)
                                if fusion_config and 'fusion_weights' in fusion_config:
                                    alpha = fusion_config['fusion_weights'].get('alpha', 0.5)
                                    beta = fusion_config['fusion_weights'].get('beta', 0.5)
                                    # 验证权重和是否为1（允许0.001的误差）
                                    if abs(alpha + beta - 1.0) > 0.001:
                                        print(f"警告：权重和不为1，alpha={alpha}, beta={beta}")
                                        # 归一化权重
                                        total = alpha + beta
                                        alpha = alpha / total
                                        beta = beta / total
                                else:
                                    # 如果配置不存在，使用默认权重
                                    alpha = 0.5
                                    beta = 0.5
                                    print("警告：未找到融合权重配置，使用默认权重0.5")
                                
                                llm_suggested_confidence = llm_result.get('llm_suggested_confidence', dnn_confidence)
                                
                                # 强制使用加权平均公式，确保测试集不使用confidence_change公式
                                final_confidence = alpha * dnn_confidence + beta * llm_suggested_confidence
                                
                                # 添加调试信息，验证计算过程
                                print(f"测试集加权平均计算: {dnn_confidence:.4f}×{alpha:.4f} + {llm_suggested_confidence:.4f}×{beta:.4f} = {final_confidence:.4f}")
                                
                                debug_info = f"测试集加权平均: {dnn_confidence:.4f}×{alpha:.4f} + {llm_suggested_confidence:.4f}×{beta:.4f} = {final_confidence:.4f}"
                            else:
                                # 训练集：使用llm_validator.py中的调整逻辑
                                # 关键修复：训练集模式下，即使使用缓存，也要基于当前dnn_confidence重新计算final_confidence
                                # 而不是直接使用缓存中的final_confidence值
                                
                                # 获取LLM判断和建议的置信度
                                llm_judgment = llm_result.get('llm_judgment', '否')
                                llm_suggested_confidence = llm_result.get('llm_suggested_confidence', dnn_confidence)
                                
                                # 使用llm_validator.py中的调整公式重新计算
                                confidence_change = abs(llm_suggested_confidence - dnn_confidence)
                                
                                if llm_judgment == "否":
                                    # LLM判断为"否"时，置信度应该降低
                                    final_confidence = max(0.1, dnn_confidence - confidence_change * 0.8)
                                elif llm_judgment == "是":
                                    # LLM判断为"是"时，置信度应该升高
                                    final_confidence = min(0.9, dnn_confidence + confidence_change * 0.8)
                                else:
                                    # 其他情况使用原始置信度
                                    final_confidence = dnn_confidence
                                
                                # 添加调试信息，显示重新计算过程
                                debug_info = f"训练集LLM调整(重新计算): {dnn_confidence:.4f}→{final_confidence:.4f} (判断:{llm_judgment}, 建议:{llm_suggested_confidence:.4f})"
                                
                                # 如果是缓存结果，添加缓存标记
                                if strategy and hasattr(strategy, 'llm_cache') and cache_key in strategy.llm_cache:
                                    debug_info += " [缓存结果重新计算]"
                            
                            llm_judgment = llm_result.get('llm_judgment', llm_result.get('error', '解析失败'))
                            llm_reason = llm_result.get('reason', llm_result.get('api_failure_reason', 'LLM响应格式不符合预期'))
                            
                            # 根据置信度阈值修正预测：final_confidence>=0.5为有坏味，final_confidence<0.5为无坏味
                            final_prediction = 1 if final_confidence >= 0.5 else 0
                            
                            was_corrected = final_prediction != dnn_prediction
                            
                            # 添加llm_suggested_confidence字段
                            llm_suggested_confidence = llm_result.get('llm_suggested_confidence')
                    else:
                        failure_reason = llm_result.get('api_failure_reason', llm_result.get('error', '未知原因')) if llm_result else "LLM结果为空"
                        llm_judgment = "验证失败"
                        llm_reason = failure_reason
                        llm_suggested_confidence = None
                        debug_info = f"LLM验证失败: {failure_reason}"
                else:
                    # 代码片段获取失败
                    final_prediction = dnn_prediction
                    final_confidence = dnn_confidence
                    llm_judgment = "验证失败"
                    llm_reason = "代码片段获取失败"
                    was_corrected = False
                    
                    debug_info = "代码片段获取失败，跳过LLM验证"
            else:
                # LLM未介入
                debug_info = decision_reason
            
            # 创建结果记录
            result = {
                'entity_key': entity_key,
                'smell_type': self.smell_type,
                'dnn_confidence': dnn_confidence,
                'dnn_prediction': dnn_prediction,
                'final_confidence': final_confidence,
                'final_prediction': final_prediction,
                'was_corrected': was_corrected,
                'true_label': data.get('label', 0),
                'debug_info': debug_info,
                'decision_strategy': decision_reason,
                'llm_enhanced': needs_llm_validation
            }
            
            # 如果LLM介入，添加相关字段
            if needs_llm_validation:
                result['llm_judgment'] = llm_judgment
                result['llm_reason'] = llm_reason
                
                # 添加llm_suggested_confidence字段（如果有）
                if llm_suggested_confidence is not None:
                    result['llm_suggested_confidence'] = llm_suggested_confidence
            
            # 添加was_corrected字段的详细说明
            result['was_corrected_explanation'] = "was_corrected表示LLM是否改变了DNN的原始预测结果：true表示LLM修正了DNN的预测，false表示LLM保持了DNN的原始预测"
            results.append(result)
            
            # 如果使用策略类，添加观察数据
            if strategy:
                strategy.add_observation(
                    confidence=dnn_confidence,
                    dnn_pred=dnn_prediction,
                    llm_pred=final_prediction if needs_llm_validation else None,
                    true_label=data.get('label', None),
                    was_intervened=needs_llm_validation
                )
            
            # 显示示例（如果启用）
            if show_examples > 0 and i < show_examples:
                print(f"示例 {i+1}: {entity_key}")
                print(f"  DNN预测: {dnn_prediction}, 置信度: {dnn_confidence:.3f}")
                print(f"  最终预测: {final_prediction}, 置信度: {final_confidence:.3f}")
                print(f"  决策策略: {decision_reason}")
                if needs_llm_validation:
                    print(f"  LLM介入: {llm_judgment}")
                print()
        
        return {
            'results': results
        }

    def predict_single(self, data: Dict[str, Any]) -> Dict[str, float]:
        """对单个样本进行DNN预测
        
        Args:
            data: 包含metrics和code_embedding的字典
            
        Returns:
            包含prediction和confidence的字典
        """
        try:
            metrics = data['metrics']
            code_embedding = data['code_embedding']
            
            # 转换为张量
            metrics_tensor = torch.FloatTensor([metrics]).to(self.device)
            code_tensor = torch.FloatTensor([code_embedding]).to(self.device)
            
            # DNN预测
            with torch.no_grad():
                output = self.dnn_model(metrics_tensor, code_tensor)
                probabilities = torch.softmax(output, dim=1)
                
                # 获取预测结果和置信度
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities.max().item()
                
                # 对于二分类，如果是类别1，返回对应的概率
                if probabilities.shape[1] == 2:
                    # 返回类别1的概率作为坏味存在的可能性
                    confidence = probabilities[0][1].item()
                    prediction = 1 if confidence >= self.threshold else 0
                
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

    def should_use_llm_validation(self, dnn_confidence: float, dnn_prediction: int, smell_type: str) -> bool:
        """判断是否需要LLM验证
        
        Args:
            dnn_confidence: DNN预测的置信度
            dnn_prediction: DNN预测的结果
            smell_type: 坏味类型
            
        Returns:
            是否需要LLM验证的布尔值
        """
        # 基于置信度的简单决策逻辑
        # 当置信度在[0.3, 0.7)区间时，需要LLM验证
        return 0.3 <= dnn_confidence < 0.7

    def _load_fusion_config(self, smell_type: str, fold_idx: int = None) -> Dict:
        """加载融合配置，根据坏味类型和fold信息选择配置文件
        
        Args:
            smell_type: 坏味类型
            fold_idx: 折数索引（五折交叉验证时使用）
        """
        # 优先尝试加载包含fold信息的配置文件
        if fold_idx is not None:
            config_path = f"results/{smell_type}_fold{fold_idx}_config.json"
            if os.path.exists(config_path):
                print(f"[OK] 使用五折交叉验证配置文件: {config_path}")
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception as e:
                    print(f"[ERROR] 加载五折交叉验证配置文件失败: {e}")
        
        # 如果五折交叉验证配置文件不存在，尝试通用配置文件
        config_path = f"results/{smell_type}_fusion_config.json"
        
        # 不同坏味类型的默认配置
        default_configs = {
            "long_method": {
                "risk_zones": [
                    {"range": "0.48-0.51", "requires_llm": True},
                    {"range": "0.51-0.54", "requires_llm": True},
                    {"range": "0.54-0.57", "requires_llm": True},
                    {"range": "0.57-0.60", "requires_llm": True},
                    {"range": "0.60-0.63", "requires_llm": True},
                    {"range": "0.63-0.66", "requires_llm": True}
                ],
                "fusion_weights": {
                    "alpha": 0.3626,
                    "beta": 0.6374
                }
            },
            "feature_envy": {
                "risk_zones": [
                    {"range": "0.3-0.7", "requires_llm": True}
                ],
                "fusion_weights": {
                    "alpha": 0.5,
                    "beta": 0.5
                }
            },
            "blob": {
                "risk_zones": [
                    {"range": "0.3-0.7", "requires_llm": True}
                ],
                "fusion_weights": {
                    "alpha": 0.5,
                    "beta": 0.5
                }
            },
            "data_class": {
                "risk_zones": [
                    {"range": "0.3-0.7", "requires_llm": True}
                ],
                "fusion_weights": {
                    "alpha": 0.5,
                    "beta": 0.5
                }
            }
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                print(f"⚠️ 融合配置文件不存在: {config_path}")
                # 返回该坏味类型的默认配置
                default_config = default_configs.get(smell_type, default_configs["long_method"])
                print(f"📋 使用{smell_type}的默认融合配置")
                return default_config
        except Exception as e:
            print(f"❌ 加载融合配置失败: {e}")
            # 返回通用默认配置
            return {
                "risk_zones": [{"range": "0.3-0.7", "requires_llm": True}],
                "fusion_weights": {"alpha": 0.5, "beta": 0.5}
            }


def load_test_data_for_smell(smell_type: str, model_path: str = None, fold_idx: int = None, data_type: str = "training") -> Tuple[List[Dict], Dict]:
    """加载特定坏味类型的训练数据，支持五折交叉验证"""
    
    # 设置固定随机种子，确保数据划分可复现
    seed = 666
    random.seed(seed)
    
    # 加载标签 - 使用数据集标签
    label_file = f"dataset/labels_{smell_type}.txt"
    with open(label_file, 'r') as f:
        all_lines = f.readlines()
    
    # 打乱标签列表，与deepLearningDetectionX5.py保持一致
    random.shuffle(all_lines)
    
    # 加载各种指标数据 - 使用数据集数据
    all_data_dict = getDictByJson("dataset/allDataDict.json")
    allCommonMetrics = getDictByJson("dataset/allCommonMetrics.json")
    allCommitMetrics = getDictByJson("dataset/allCommitMetrics.json")  # 添加commit metrics
    allStructuralMetrics = getDictByJson("dataset/allStructuralMetrics.json")
    finalSelectedSynMetrics = getDictByJson("dataset/finalSelectedSynMetrics.json")  # 添加syntax metrics
    allSemanticEdges = getDictByJson("dataset/allSemanticEdges.json")
    allCodeEmbDict = getDictByJson("dataset/allCodeVectors.json")
    
    # 五折交叉验证数据划分
    if fold_idx is not None:
        # 按严重程度分层抽样进行五折交叉验证划分
        fold_num = 5
        
        # 按严重程度分层
        none_item = []
        minor_item = []
        major_item = []
        critical_item = []
        
        for item in all_lines:
            try:
                deg = int(item.rstrip('\n').split()[2])
                if deg == 0:
                    none_item.append(item)
                elif deg == 1:
                    minor_item.append(item)
                elif deg == 2:
                    major_item.append(item)
                elif deg == 3:
                    critical_item.append(item)
            except:
                continue
        
        # 对每个严重程度类别进行分层抽样
        def getTypeSplitList(type_list, fold_num, fold_idx):
            """将指定类型的样本列表按折数划分"""
            fold_size = len(type_list) // fold_num
            start_index = (fold_idx - 1) * fold_size
            end_index = fold_idx * fold_size
            train_list = type_list[:start_index] + type_list[end_index:]
            test_list = type_list[start_index:end_index]
            return train_list, test_list
        
        none_train, none_test = getTypeSplitList(none_item, fold_num, fold_idx)
        minor_train, minor_test = getTypeSplitList(minor_item, fold_num, fold_idx)
        major_train, major_test = getTypeSplitList(major_item, fold_num, fold_idx)
        critical_train, critical_test = getTypeSplitList(critical_item, fold_num, fold_idx)

        # 合并训练集和测试集
        if data_type == "training":
            selected_lines = none_train + minor_train + major_train + critical_train
        else:  # testing
            selected_lines = none_test + minor_test + major_test + critical_test
            
        # 确保每个fold中至少包含一个样本
        for severity in [0, 1, 2, 3]:  # none, minor, major, critical
            if len([x for x in selected_lines if int(x.split()[2]) == severity]) == 0:
                # 如果测试集中缺少某个类别，从训练集中移动一个样本
                other_lines = none_test + minor_test + major_test + critical_test if data_type == "training" else none_train + minor_train + major_train + critical_train
                severity_samples = [x for x in other_lines if int(x.split()[2]) == severity]
                if severity_samples:
                    selected_lines.append(severity_samples[0])
    else:
        # 不使用交叉验证，使用所有数据
        selected_lines = all_lines
    
    test_data = []
    for line in selected_lines:
        try:
            info = line.rstrip('\n').split()
            entity_key = info[0]
            code_name = entity_key + ".java"
            label = int(info[1])
            
            # 构建特征向量（与deepLearningDetectionX5.py保持一致）
            # 使用dict[key]直接访问，缺失时抛出异常被跳过
            metrics = allCommonMetrics[entity_key] + allCommitMetrics[entity_key] + \
                      allStructuralMetrics[entity_key] + finalSelectedSynMetrics[entity_key] + \
                      allSemanticEdges[entity_key]
            
            code_embedding = allCodeEmbDict[code_name]
            
            test_data.append({
                "entity_key": entity_key,
                "metrics": metrics,
                "code_embedding": code_embedding,
                "label": label,
                "true_label": label  # 添加真实标签用于性能评估
            })
                
        except:
            # 特征缺失时跳过该样本（与deepLearningDetectionX5.py一致）
            pass
    
    return test_data, all_data_dict

# 复制工具函数
def getDictByJson(jsonPath):
    with open(jsonPath, 'r') as f:
        mydata = f.read()
    return json.loads(mydata)



if __name__ == "__main__":
    # 使用示例 - 传统模式（不使用交叉验证）
    model_path = "model/MyDNN_fusion_feature_envy_fold1_binary.pth"
    detector = EnhancedCodeSmellDetector(model_path)
    
    # 测试特定坏味类型
    smell_type = "feature_envy"
    test_data, all_data_dict = load_test_data_for_smell(smell_type, model_path)
    
    results = detector.batch_predict_enhanced(
        test_data=test_data,
        all_data_dict=all_data_dict,
        smell_type=smell_type,
        is_testing=True  # 测试集使用加权平均计算final_confidence
    )
    
    # 输出统计
    llm_enhanced_count = sum(1 for r in results if r["llm_enhanced"])
    print(f"LLM增强案例数: {llm_enhanced_count}/{len(results)}")
    
    # 保存结果
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/enhanced_{smell_type}_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("✅ 传统模式检测完成")
    
    # 使用示例 - 五折交叉验证模式
    print("\n🎯 五折交叉验证模式示例:")
    for fold_idx in range(1, 6):
        print(f"\n📊 运行Fold {fold_idx}...")
        
        # 创建检测器（支持五折交叉验证）
        detector_cv = EnhancedCodeSmellDetector(
            model_path=model_path,
            fold_idx=fold_idx
        )
        
        # 加载测试数据（支持五折交叉验证）
        test_data_cv, all_data_dict_cv = load_test_data_for_smell(
            smell_type=smell_type, 
            model_path=model_path,
            fold_idx=fold_idx,
            data_type="testing"
        )
        
        # 批量预测（支持五折交叉验证）
        results_cv = detector_cv.batch_predict_enhanced(
            test_data=test_data_cv,
            all_data_dict=all_data_dict_cv,
            smell_type=smell_type,
            fold_idx=fold_idx,
            is_testing=True  # 测试集使用加权平均计算final_confidence
        )
        
        print(f"✅ Fold {fold_idx} 完成，处理样本数: {len(results_cv)}")
