# 大模型评测框架

自制大模型评测系统，实现三种打分 + 两模型对比 + 可视化报告。

## 技术栈

`PyTorch` · `Qwen2.5` · `sentence-transformers` · `LLM-as-Judge` · `matplotlib`

## 项目结构

```
├── evaluate.py          # 评测主脚本
├── eval_dataset.json    # 20题评测集（含标准答案+关键词）
├── plot_results.py      # 可视化脚本
├── eval_results.json    # 评测结果数据
└── eval_report.png      # 可视化报告
```

## 评测方法

| 打分方式 | 说明 |
|---|---|
| 关键词匹配 | 模型回答命中标准答案关键词的比例 |
| 语义相似度 | 用 sentence-transformers 计算回答与标准答案的语义距离 |
| LLM-as-Judge | 用大模型从准确性/完整性/流畅性/安全性四维度 1-5 打分 |

## 评测结果（原始 vs 微调后 Qwen2.5-1.5B）

| 指标 | 原始模型 | 微调模型 | 提升 |
|---|---|---|---|
| 关键词命中率 | 0.425 | 0.442 | +0.017 |
| 语义相似度 | 0.724 | 0.767 | +0.043 |
| LLM-Judge | 4.76 | 5.54 | +0.78 |
