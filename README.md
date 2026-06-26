# CF-Fusion

CF-Fusion is a replication package for confidence-driven DNN-LLM collaborative code smell detection.

## English

### Included

- `dataset/*.json`: extracted features and metadata
- `dataset/labels_*.txt`: labels
- `fold_splits/`: five-fold split indexes
- `results/` and `results_weighted/`: fold outputs and configs
- `thresholds_weighted/`: fold-specific thresholds

### Excluded

- `dataset/sourceCode/`: raw Java source snippets
- large caches, checkpoints, and other disposable artifacts

The raw source snippets come from the public MLCQ dataset. To run the LLM path locally, download the original data from its official source and restore the source snippets under `dataset/sourceCode/`.

### Setup

```bash
pip install -r requirements.txt
```

Set the API key in your local environment before running LLM-based scripts:

```powershell
$env:SILICONFLOW_API_KEY="<your_siliconflow_api_key>"
```

### Reproduction

Train the DNN:

```bash
python deepLearningDetectionX5.py --smell feature_envy
```

Generate weighted configs:

```bash
python generate_config_weighted.py --smell-type feature_envy --fold 1 --results-dir results_weighted
```

Run all folds in PowerShell:

```powershell
1..5 | ForEach-Object { python generate_config_weighted.py --smell-type feature_envy --fold $_ --results-dir results_weighted }
```

The weighted test code reads:

```text
results_weighted/{smell_type}_fold{fold_idx}_config.json
```

`alpha` and `beta` are fold-specific fusion weights selected on the training split.

### Notes

- `cold_start_excluded: 50` means 50 benchmark samples were excluded from the fusion-weight search.
- The API key is read from `SILICONFLOW_API_KEY`.

## 中文

### 包含内容

- `dataset/*.json`：特征与元数据
- `dataset/labels_*.txt`：标签
- `fold_splits/`：五折划分索引
- `results/` 和 `results_weighted/`：各 fold 的输出与配置
- `thresholds_weighted/`：逐 fold 阈值

### 不包含内容

- `dataset/sourceCode/`：原始 Java 源码片段
- 大体积缓存、checkpoint 和其他临时产物

原始源码片段来自公开的 MLCQ 数据集。若要本地运行 LLM 流程，请从官方来源下载原始数据，并将源码片段恢复到 `dataset/sourceCode/`。

### 环境

```bash
pip install -r requirements.txt
```

在本地运行 LLM 相关脚本前，请先设置环境变量：

```powershell
$env:SILICONFLOW_API_KEY="<your_siliconflow_api_key>"
```

### 复现

训练 DNN：

```bash
python deepLearningDetectionX5.py --smell feature_envy
```

生成加权配置：

```bash
python generate_config_weighted.py --smell-type feature_envy --fold 1 --results-dir results_weighted
```

PowerShell 一次生成全部 fold：

```powershell
1..5 | ForEach-Object { python generate_config_weighted.py --smell-type feature_envy --fold $_ --results-dir results_weighted }
```

加权测试代码读取：

```text
results_weighted/{smell_type}_fold{fold_idx}_config.json
```

`alpha` 和 `beta` 是在训练集上选出的逐 fold 融合权重。

### 说明

- `cold_start_excluded: 50` 表示有 50 个标杆样本不参与融合权重搜索。
- API key 由 `SILICONFLOW_API_KEY` 读取。
