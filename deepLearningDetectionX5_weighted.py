from mimetypes import common_types
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm, trange
import argparse
import random
import json
import time
import os
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

from deepLearningModuleX5 import *

from focalloss import FocalLossMulti, FocalLoss

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--d_model', type=int, default=128)
parser.add_argument('--num_encoder_layers', type=int, default=3)
parser.add_argument('--dim_feedforward', type=int, default=512)
parser.add_argument('--dropout', type=int, default=0.1)
parser.add_argument('--hidden_dropout_prob', type=int, default=0.1)
parser.add_argument('--attention_probs_dropout_prob', type=int, default=0.1)
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--num_attention_heads', type=int, default=8)
parser.add_argument('--alpha', type=int, default=0.2)
parser.add_argument("--threshold", default=0)
parser.add_argument('--smell', type=str, default='all', 
                    choices=['all', 'data_class', 'blob', 'feature_envy', 'long_method'],
                    help='指定要训练的代码坏味类型，默认为all（训练所有类型）')
args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

criterion = FocalLoss().to(device)


def getDictByJson(jsonPath):
    with open(jsonPath, 'r') as f:
        mydata = f.read()
    mydict = json.loads(mydata)
    return mydict






def getTypeSplitList(type_list, fold_num, fold_idx):
    """将指定类型的样本列表按折数划分"""
    fold_size = len(type_list) // fold_num
    start_index = (fold_idx - 1) * fold_size
    end_index = fold_idx * fold_size
    train_list = type_list[:start_index] + type_list[end_index:]
    test_list = type_list[start_index:end_index]
    return train_list, test_list


def getTrainAndTestSetBySeedFold(label_list, fold_num, fold_idx):
        fold_size = len(label_list)//fold_num
        print("fold_size",fold_size)
        none_item = []
        minor_item = []
        major_item = []
        critical_item = []
        for item in label_list:
            try:
                deg = int(item.rstrip('\n').split()[2])
                if deg == 0:
                    none_item.append(item)
                if deg == 1:
                    minor_item.append(item)
                if deg == 2:
                    major_item.append(item)
                if deg == 3:
                    critical_item.append(item)
            except:
                continue

        none_train, none_test = getTypeSplitList(none_item, fold_num, fold_idx)
        minor_train, minor_test = getTypeSplitList(minor_item, fold_num, fold_idx)
        major_train, major_test = getTypeSplitList(major_item, fold_num, fold_idx)
        critical_train, critical_test = getTypeSplitList(critical_item, fold_num, fold_idx)

        test_label_list = none_test + minor_test + major_test + critical_test
        train_label_list = none_train + minor_train + major_train + critical_train

        return train_label_list, test_label_list






def showRitiaOfPosNeg(train_label_list, test_label_list):
    train_pos = 0
    train_neg = 0
    test_pos = 0
    test_neg = 0
    for item in train_label_list:
        label = int(item.split()[1])
        if label == 1:
            train_pos += 1
        else:
            train_neg += 1

    for item in test_label_list:
        label = int(item.split()[1])
        if label == 1:
            test_pos += 1
        else:
            test_neg += 1
    print('showRitiaOfPosNeg:', train_pos / train_neg, test_pos / test_neg)


def Undersampling(trainlist):
    print('trainlist', len(trainlist))
    # 使用当前时间作为随机种子，每个epoch采样不同数据
    seed = int(time.time())
    random.seed(seed)
    random.shuffle(trainlist)
    pos = 0
    neg = 0
    posSamples = []
    negSamples = []
    selectSamples = []
    for sample in trainlist:
        if sample.split()[1] == '0':
            neg += 1
            negSamples.append(sample)
        else:
            pos += 1
            posSamples.append(sample)
    print('sample ratio(pos:neg): ', pos, ':', neg)
    if pos >= neg:
        selectSamples = negSamples + posSamples[:neg]
    elif neg > pos:
        selectSamples = negSamples[:pos] + posSamples
    # 使用相同的随机种子
    random.seed(seed)
    random.shuffle(selectSamples)
    pos = 0
    neg = 0
    for item in selectSamples:
        if item.split()[1] == '0':
            neg += 1
        else:
            pos += 1
    print('after sampling(pos:neg): ', pos, ':', neg)
    return selectSamples


def getBatchList(our_RQ, allCommonMetrics, allCommitMetrics, allStructuralMetrics, finalSelectedSynMetrics,
                 allSemanticEdges, allCodeEmbDict, line_list):
    batchData = []
    batch_metric = []
    batch_codeEmbedding = []
    batch_label = []
    for line in line_list:
        try:
            info = line.rstrip('\n').split()
            # print('info',info)
            codeName = info[0] + ".java"
            label = int(info[1])

            if our_RQ == 1 or our_RQ == 2:
                metrics = allCommonMetrics[info[0]] + allCommitMetrics[info[0]] + allStructuralMetrics[info[0]] + \
                          finalSelectedSynMetrics[info[0]] + allSemanticEdges[info[0]]
                # metrics = allStructuralMetrics[info[0]] + finalSelectedSynMetrics[info[0]] + allSemanticEdges[info[0]]
            elif our_RQ == 3:
                metrics = allCommonMetrics[info[0]] + allCommitMetrics[info[0]]
            elif our_RQ == 4 and metric_type == metricType[
                0]:  # metricType = ['common', 'commit', 'structure', 'syntax', 'semantics']
                metrics = allCommonMetrics[info[0]]
            elif our_RQ == 4 and metric_type == metricType[1]:
                metrics = allCommitMetrics[info[0]]
            elif our_RQ == 4 and metric_type == metricType[2]:
                metrics = allStructuralMetrics[info[0]]
            elif our_RQ == 4 and metric_type == metricType[3]:
                metrics = finalSelectedSynMetrics[info[0]]
            elif our_RQ == 4 and metric_type == metricType[4]:
                metrics = allSemanticEdges[info[0]]

            codeEmbedding = allCodeEmbDict[codeName]
            batch_metric.append(metrics)
            batch_codeEmbedding.append(codeEmbedding)
            batch_label.append(label)
            batchData.append([batch_metric, batch_codeEmbedding, batch_label])
        except:
            pass
    return batchData


def getBatch(line_list, batch_size, batch_index, our_RQ, device):
    start_line = batch_size * batch_index
    end_line = start_line + batch_size
    batchData = getBatchList(our_RQ, allCommonMetrics, allCommitMetrics, allStructuralMetrics, finalSelectedSynMetrics,
                             allSemanticEdges, allCodeEmbDict, line_list[start_line:end_line])
    return batchData


def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list


def find_best_threshold(y_true, y_prob, metric='f1'):
    """寻找最优分类阈值
    Args:
        y_true: 真实标签
        y_prob: 预测概率（正类）
        metric: 优化指标，可选 'f1', 'precision', 'recall', 'accuracy'
    Returns:
        best_threshold: 最优阈值
        best_metric_value: 最优指标值
    """
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_metric_value = 0.0
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        if metric == 'f1':
            value = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        elif metric == 'precision':
            value = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        elif metric == 'recall':
            value = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        elif metric == 'accuracy':
            value = accuracy_score(y_true, y_pred)
        else:
            value = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        if value > best_metric_value:
            best_metric_value = value
            best_threshold = threshold
    
    return best_threshold, best_metric_value


def test(testlist, model_index, model=None, use_best_threshold=False):
    # 预测
    if model is None:
        model = globals().get('model')  # 如果没有传入模型，使用全局变量
    
    model.eval()

    batchData = getBatch(testlist, 100000, 0, our_RQ, device)
    batch_metric, batch_codeEmbedding, batch_label = batchData[0]

    x = torch.as_tensor(batch_metric).to(device)
    y = torch.as_tensor(batch_codeEmbedding).to(device)
    batch_label = torch.LongTensor(batch_label).to(device)

    with torch.no_grad():
        y_pred = model(x, y)
        y_scores = y_pred.cpu().numpy()
    
    y_test = batch_label.cpu()
    
    # 计算概率
    y_prob = torch.softmax(torch.tensor(y_scores), dim=1).numpy()[:, 1]
    
    # 寻找最优阈值（基于F1分数）
    if use_best_threshold:
        best_threshold, _ = find_best_threshold(y_test.numpy(), y_prob, metric='f1')
        print(f'最优阈值: {best_threshold:.2f}')
        y_pred_label = (y_prob >= best_threshold).astype(int)
    else:
        # 使用默认0.5阈值
        y_pred_label = np.argmax(y_scores, axis=1)
    
    # 计算p, r, f1, acc
    p = precision_score(y_test, y_pred_label, average='weighted', zero_division='warn')
    r = recall_score(y_test, y_pred_label, average='weighted')
    f1 = f1_score(y_test, y_pred_label, average='weighted')
    acc = accuracy_score(y_test, y_pred_label)

    # 计算auc
    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception as e:
        print(f"Warning: Binary AUC calculation failed: {str(e)}")
        auc = 0.0

    # 输出指标
    print('Precision: {:.4f}'.format(p))
    print('Recall: {:.4f}'.format(r))
    print('F1: {:.4f}'.format(f1))
    print('Accuracy: {:.4f}'.format(acc))
    print('AUC: {:.4f}'.format(auc))

    # 确保所有指标都有有效值
    p = float(format(p, '.4f'))
    r = float(format(r, '.4f'))
    f1 = float(format(f1, '.4f'))
    acc = float(format(acc, '.4f'))
    auc = float(format(auc, '.4f'))

    print("\n p, r, f1, acc, auc:", p, r, f1, acc, auc)
    return p, r, f1, acc, auc


def train_single_fold(model, train_label_list, test_label_list, fold_idx, test_result):
    """训练单个fold的模型"""
    # 初始化最大值变量
    p_max = r_max = f1_max = acc_max = auc_max = 0.0
    
    # 记录最佳F1对应的epoch（与RQ5_OurApproach.py一致，无早停机制）
    iterations = 0

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()

    # 记录当前fold信息
    fold_index_record = open(test_result, 'a')
    fold_index_record.write(f"\n\n-------Fold {fold_idx} Testing Results--------\n")
    fold_index_record.close()

    # 断点重续检查 - 为每个fold创建独立的断点文件
    start_epoch = 0
    model_dir = os.path.join(PROJECT_ROOT, "model_weighted")
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(
        model_dir, f"{curModel.__name__}_{smell}_fold{fold_idx}_binary.pth"
    )
    checkpoint_path = os.path.join(
        model_dir, f"{curModel.__name__}_{smell}_fold{fold_idx}_binary_checkpoint.pth"
    )
    
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            p_max = checkpoint.get('p_max', 0.0)
            r_max = checkpoint.get('r_max', 0.0)
            f1_max = checkpoint.get('f1_max', 0.0)
            acc_max = checkpoint.get('acc_max', 0.0)
            auc_max = checkpoint.get('auc_max', 0.0)
            print(f"Fold {fold_idx}: 从断点恢复训练，从epoch {start_epoch} 开始")
            print(f"Fold {fold_idx}: 当前最佳指标 - F1: {f1_max:.4f}, AUC: {auc_max:.4f}")
        except Exception as e:
            print(f"Fold {fold_idx}: 断点恢复失败: {str(e)}，重新开始训练")
            start_epoch = 0

    iterations = 0
    epochs = trange(start_epoch, args.epochs, leave=True, desc=f"Fold {fold_idx} Epoch")
    for epoch in epochs:
        totalloss = 0.0
        main_index = 0.0
        count = 0
        right = 0
        acc = 0

        # 使用随机欠采样（与binary版本一致）
        trainlist = Undersampling(train_label_list)
        model.train()
        for batch_index in tqdm(range(int(len(trainlist) / args.batch_size))):
            optimizer.zero_grad()
            batchData = getBatch(trainlist, args.batch_size, batch_index, our_RQ, device)
            batch_metric, batch_codeEmbedding, batch_label = batchData[0]

            x = torch.as_tensor(batch_metric).to(device)
            y = torch.as_tensor(batch_codeEmbedding).to(device)
            batch_label = F.one_hot(torch.LongTensor(batch_label), 2).to(device)

            output = model(x, y)
            batchloss = criterion(output, batch_label.float())
            right += torch.sum(torch.eq(torch.argmax(output, dim=1), torch.argmax(batch_label, dim=1)))

            count += len(batch_metric)
            acc = right * 1.0 / count
            batchloss.backward(retain_graph=True)
            optimizer.step()
            loss = batchloss.item()
            totalloss += loss
            main_index = main_index + len(batch_metric)
            loss = totalloss / main_index
            epochs.set_description(f"Fold {fold_idx} Epoch (Loss=%.6g) (Acc = %.6g)" % (round(loss, 5), acc))
            iterations += 1

        p, r, f1, acc, auc = test(test_label_list, epoch, model)

        # 更新最大值 - 使用AUC作为主要指标（与binary版本一致）
        if auc > auc_max:
            p_max = p
            r_max = r
            f1_max = f1
            acc_max = acc
            auc_max = auc
            iterations = epoch
            test_p_r_f1 = open(test_result, 'a')
            test_p_r_f1.write(
                f'fold{fold_idx}_epoch{epoch} {p} {r} {f1} {acc} {auc}\n')
            test_p_r_f1.close()
            
            # 保存最佳模型
            torch.save(model.state_dict(), best_model_path)
            print(f"Fold {fold_idx}: 最佳模型已保存: {best_model_path} (AUC={auc:.4f})")

        # 每5个epoch保存一次断点
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'p_max': p_max,
                'r_max': r_max,
                'f1_max': f1_max,
                'acc_max': acc_max,
                'auc_max': auc_max
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Fold {fold_idx}: 断点已保存: epoch {epoch + 1}")

    # 训练完成后删除断点文件
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Fold {fold_idx}: 训练完成，断点文件已清理")
    
    return p_max, r_max, f1_max, acc_max, auc_max


def train_cross_validation(model, smell_type, test_result, label_list, fold_num):
    """五折交叉验证训练"""
    all_metrics = []
    
    # 记录交叉验证开始信息
    cv_record = open(test_result, 'a')
    cv_record.write("\n\n======= 5-Fold Cross Validation =======\n")
    cv_record.close()
    
    for fold_idx in range(1, fold_num + 1):
        print(f"\n{'='*50}")
        print(f"开始训练 Fold {fold_idx}/{fold_num}")
        print(f"{'='*50}")
        
        # 获取当前fold的训练集和测试集
        train_label_list, test_label_list = getTrainAndTestSetBySeedFold(label_list, fold_num, fold_idx)
        print(f"Fold {fold_idx}: 训练集大小: {len(train_label_list)}, 测试集大小: {len(test_label_list)}")
        showRitiaOfPosNeg(train_label_list, test_label_list)
        
        # 为当前fold创建新的模型实例
        fold_model = model.__class__(model.metricSize, args.hidden, 2).to(device)
        
        # 训练当前fold
        p, r, f1, acc, auc = train_single_fold(fold_model, train_label_list, test_label_list, fold_idx, test_result)
        
        # 记录当前fold的结果
        all_metrics.append({
            'fold': fold_idx,
            'precision': p,
            'recall': r,
            'f1': f1,
            'accuracy': acc,
            'auc': auc
        })
        
        # 记录当前fold结果
        cv_record = open(test_result, 'a')
        cv_record.write(f"Fold {fold_idx}: P={p:.4f}, R={r:.4f}, F1={f1:.4f}, Acc={acc:.4f}, AUC={auc:.4f}\n")
        cv_record.close()
        
        print(f"Fold {fold_idx} 完成: P={p:.4f}, R={r:.4f}, F1={f1:.4f}, Acc={acc:.4f}, AUC={auc:.4f}")
    
    # 计算平均指标
    avg_p = np.mean([m['precision'] for m in all_metrics])
    avg_r = np.mean([m['recall'] for m in all_metrics])
    avg_f1 = np.mean([m['f1'] for m in all_metrics])
    avg_acc = np.mean([m['accuracy'] for m in all_metrics])
    avg_auc = np.mean([m['auc'] for m in all_metrics])
    
    # 记录平均结果
    cv_record = open(test_result, 'a')
    cv_record.write(f"\n平均结果: P={avg_p:.4f}, R={avg_r:.4f}, F1={avg_f1:.4f}, Acc={avg_acc:.4f}, AUC={avg_auc:.4f}\n")
    cv_record.write("======= 5-Fold Cross Validation 完成 =======\n")
    cv_record.close()
    
    print(f"\n{'='*50}")
    print("五折交叉验证完成")
    print(f"平均结果: P={avg_p:.4f}, R={avg_r:.4f}, F1={avg_f1:.4f}, Acc={avg_acc:.4f}, AUC={avg_auc:.4f}")
    print(f"{'='*50}")
    
    return avg_p, avg_r, avg_f1, avg_acc, avg_auc


if __name__ == '__main__':
    # 设置固定随机种子，确保实验可复现性
    seed = 666
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"使用固定随机种子: {seed}")

    smellType = ['data_class', 'blob', 'feature_envy', 'long_method']
    metricType = ['common', 'commit', 'structure', 'syntax', 'semantics']

    # javaMetricsPP metrics ## 16 / 4 common metrics  +  19 commit metrics
    dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
    results_dir = os.path.join(PROJECT_ROOT, "results_weighted")
    os.makedirs(results_dir, exist_ok=True)

    allCommonMetricsPath = os.path.join(dataset_dir, "allCommonMetrics.json")
    allCommitMetricsPath = os.path.join(dataset_dir, "allCommitMetrics.json")
    allCommonMetrics = getDictByJson(allCommonMetricsPath)
    allCommitMetrics = getDictByJson(allCommitMetricsPath)

    # AST-based metrics ## 10 structureal metrics + 20 syntax metrics + 9 semantics metrics
    allStructuralMetricsPath = os.path.join(dataset_dir, "allStructuralMetrics.json")
    finalSelectedSynMetricsPath = os.path.join(dataset_dir, "finalSelectedSynMetrics.json")
    allSemanticEdgesPath = os.path.join(dataset_dir, "allSemanticEdges.json")
    allStructuralMetrics = getDictByJson(allStructuralMetricsPath)
    finalSelectedSynMetrics = getDictByJson(finalSelectedSynMetricsPath)
    allSemanticEdges = getDictByJson(allSemanticEdgesPath)

    # Code2Vec  ## 128d code embedding
    codeEmbDictPath = os.path.join(dataset_dir, "allCodeVectors.json")
    allCodeEmbDict = getDictByJson(codeEmbDictPath)

    our_RQ = 1
    # when RQ4
    metric_type = metricType[0]
    
    # 根据参数选择要训练的坏味类型
    if args.smell != 'all':
        smellType = [args.smell]
        print(f"指定训练坏味类型: {args.smell}")
    else:
        print("训练所有坏味类型: data_class, blob, feature_envy, long_method")
    
    if our_RQ == 1:  # (commonMetrics + commitMetrics + structuralMetrics + syntaxMetrics + semanticMetrics) && (codeVector)  as  model input
        modelList = [MyDNN_fusion]  # 只使用DNN
    elif our_RQ == 2:  # (commonMetrics + commitMetrics + structuralMetrics + syntaxMetrics + semanticMetrics)  or (codeVector)  as  model input
        modelList = [MyDNN_metrics, MyCNN_metrics, MyDNN_semantics, MyCNN_semantics]
    elif our_RQ == 3:  # (commonMetrics + commitMetrics) && (codeVector)  as  model input
        modelList = [MyDNN_fusion, MyCNN_fusion]
    elif our_RQ == 4:  # commonMetrics  or commitMetrics  or structuralMetrics  or syntaxMetrics  or semanticMetrics  as  model input
        modelList = [MyDNN_metrics, MyCNN_metrics]

    for curModel in modelList:
        print('model:', str(curModel))
        for smell in smellType:
            print('smell:', smell)
            ############## save experimental results ###########
            current_task = 'binary'
            num_classes = 2

            # 使用五折交叉验证模式，结果文件名添加_cv后缀
            if our_RQ == 4:
                test_result = os.path.join(
                    results_dir,
                    "result_RQ_" + str(our_RQ) + '__' + str(
                        curModel.__name__) + '_' + smell + '_' + current_task + '_' + metric_type + '_cv.txt'
                )
            else:
                test_result = os.path.join(
                    results_dir,
                    "result_RQ_" + str(our_RQ) + '__' + str(curModel.__name__) + '_' + smell + '_' + current_task + '_cv.txt'
                )
            ####################################################

            print("\n -----------------------DataInfo------------------------")
            fold_num = 5
            seed = 666
            print(smell)
            print("seed =", seed)
            print("fold_num =", fold_num)
            print("使用五折交叉验证模式")
            
            # 加载标签文件
            labelPath = os.path.join(dataset_dir, "labels_" + smell + ".txt")
            with open(labelPath) as f:
                label_list = f.readlines()
            
            # 打乱标签列表
            random.seed(seed)
            random.shuffle(label_list)
            
            # 获取数据信息用于计算metricSize
            if our_RQ == 1 or our_RQ == 2:
                if smell == "data_class" or smell == "blob":
                    metricSize = 74
                    # metricSize = 39
                elif smell == "feature_envy" or smell == "long_method":
                    metricSize = 62
                    # metricSize = 39
            elif our_RQ == 3:
                if smell == "data_class" or smell == "blob":
                    metricSize = 35
                elif smell == "feature_envy" or smell == "long_method":
                    metricSize = 23
            elif our_RQ == 4 and metric_type == metricType[
                0]:  # metricType = ['common', 'commit', 'structure', 'syntax', 'semantics']
                if smell == "data_class" or smell == "blob":
                    metricSize = 16
                elif smell == "feature_envy" or smell == "long_method":
                    metricSize = 4
            elif our_RQ == 4 and metric_type == metricType[1]:
                metricSize = 19
            elif our_RQ == 4 and metric_type == metricType[2]:
                metricSize = 10
            elif our_RQ == 4 and metric_type == metricType[3]:
                metricSize = 20
            elif our_RQ == 4 and metric_type == metricType[4]:
                metricSize = 9
            
            # 创建基础模型实例用于交叉验证
            base_model = curModel(metricSize, args.hidden, num_classes).to(device)
            
            # 执行五折交叉验证
            p, r, f1, acc, auc = train_cross_validation(base_model, smell, test_result, label_list, fold_num)
            
            test_p_r_f1 = open(test_result, 'a')
            test_p_r_f1.write(
                "\n\n五折交叉验证最终平均结果: " + str(p) + " " + str(r) + " " + str(f1) + " " + str(acc) + " " + str(
                    auc) + "\n")
            test_p_r_f1.close()
