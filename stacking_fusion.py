import json
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

def load_results(file_path):
    """加载结果文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_training_data(train_results):
    """准备训练数据（只使用有LLM介入的样本）"""
    X = []  # 特征
    y = []  # 目标

    for item in train_results.get('results', []):
        if item.get('llm_enhanced', False):
            # 提取特征
            dnn_conf = item.get('dnn_confidence', 0.5)
            llm_conf = item.get('llm_suggested_confidence', 0.5)
            
            # 添加特征
            features = [dnn_conf, llm_conf]
            X.append(features)
            
            # 提取目标
            y.append(item.get('true_label', 0))

    return np.array(X), np.array(y)

def prepare_test_data(test_results):
    """准备测试数据"""
    X_llm = []  # 有LLM介入的样本特征
    y_true = []  # 真实标签
    dnn_preds = []  # DNN原始预测
    llm_indices = []  # 有LLM介入的样本索引

    for i, item in enumerate(test_results.get('results', [])):
        # 提取真实标签
        y_true.append(item.get('true_label', 0))
        # 提取DNN原始预测
        dnn_preds.append(item.get('dnn_prediction', 0))
        
        # 检查是否有LLM介入
        if item.get('llm_enhanced', False):
            # 提取特征
            dnn_conf = item.get('dnn_confidence', 0.5)
            llm_conf = item.get('llm_suggested_confidence', 0.5)
            
            # 添加特征
            features = [dnn_conf, llm_conf]
            X_llm.append(features)
            llm_indices.append(i)

    return np.array(X_llm), np.array(y_true), np.array(dnn_preds), llm_indices

def calculate_metrics(y_true, y_pred):
    """计算性能指标"""
    return {
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_pred)
    }

def run_stacking_fusion(smell_type, total_folds=5):
    """运行Stacking融合"""
    print(f"=== 开始运行Stacking融合 for {smell_type} ===")
    
    all_metrics = []
    
    for fold in range(1, total_folds + 1):
        print(f"\n--- Processing Fold {fold} ---")
        
        # 构建文件路径
        train_file = f"results/{smell_type}_fold{fold}_train_results.json"
        test_file = f"results/{smell_type}_fold{fold}_test_results.json"
        
        # 检查文件是否存在
        if not os.path.exists(train_file):
            print(f"训练文件不存在: {train_file}")
            continue
        if not os.path.exists(test_file):
            print(f"测试文件不存在: {test_file}")
            continue
        
        # 加载数据
        train_results = load_results(train_file)
        test_results = load_results(test_file)
        
        # 准备训练数据
        X_train, y_train = prepare_training_data(train_results)
        
        if len(X_train) == 0:
            print("没有有LLM介入的训练样本")
            continue
        
        print(f"训练样本数: {len(X_train)}")
        print(f"正类样本数: {np.sum(y_train)}, 负类样本数: {len(y_train) - np.sum(y_train)}")
        
        # 训练逻辑回归模型（使用类别权重处理不平衡）
        model = LogisticRegression(random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        
        # 打印模型系数
        print(f"模型系数: {model.coef_}")
        print(f"模型截距: {model.intercept_}")
        
        # 准备测试数据
        X_test_llm, y_true, dnn_preds, llm_indices = prepare_test_data(test_results)
        
        # 生成最终预测
        final_preds = dnn_preds.copy()
        
        if len(X_test_llm) > 0:
            # 对有LLM介入的样本使用元学习器预测
            llm_preds = model.predict(X_test_llm)
            
            # 更新预测结果
            for i, idx in enumerate(llm_indices):
                final_preds[idx] = llm_preds[i]
        
        # 计算指标
        metrics = calculate_metrics(y_true, final_preds)
        all_metrics.append(metrics)
        
        print(f"Fold {fold} 指标:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    
    # 计算平均指标
    if all_metrics:
        print("\n=== 平均指标 ===")
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_value = np.mean([m[key] for m in all_metrics])
            avg_metrics[key] = avg_value
            print(f"  {key}: {avg_value:.4f}")
        
        # 保存结果
        output_file = f"results/{smell_type}_stacking_fusion_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'smell_type': smell_type,
                'total_folds': len(all_metrics),
                'average_metrics': avg_metrics,
                'fold_metrics': all_metrics
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n结果已保存到: {output_file}")
    else:
        print("没有足够的样本进行融合")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="使用Stacking融合DNN和LLM的预测结果")
    parser.add_argument('--smell-type', type=str, required=True, help="坏味类型")
    parser.add_argument('--total-folds', type=int, default=5, help="总折数")
    
    args = parser.parse_args()
    run_stacking_fusion(args.smell_type, args.total_folds)
