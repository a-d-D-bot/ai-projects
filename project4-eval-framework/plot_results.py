"""评测结果可视化"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open("eval_results.json", "r", encoding="utf-8") as f:
    results = json.load(f)

base_r = [r for r in results if r["model"] == "原始模型"]
ft_r = [r for r in results if r["model"] == "微调模型"]

# 设置中文字体（Windows 用 SimHei）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ---- 图1: 三维指标雷达图 ----
ax1 = axes[0]
metrics = ["关键词命中", "语义相似度", "LLM-Judge"]
base_vals = [
    np.mean([r["keyword_score"] for r in base_r]),
    np.mean([r["semantic_score"] for r in base_r]),
    np.mean([float(r["judge_score"]) for r in base_r]) / 5,  # 归一化到0-1
]
ft_vals = [
    np.mean([r["keyword_score"] for r in ft_r]),
    np.mean([r["semantic_score"] for r in ft_r]),
    np.mean([float(r["judge_score"]) for r in ft_r]) / 5,
]

angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]
base_vals += base_vals[:1]
ft_vals += ft_vals[:1]

ax1 = plt.subplot(1, 3, 1, projection='polar')
ax1.fill(angles, base_vals, alpha=0.25, color='#4a6cf7', label='原始模型')
ax1.plot(angles, base_vals, color='#4a6cf7', linewidth=2)
ax1.fill(angles, ft_vals, alpha=0.25, color='#da5b3b', label='微调模型')
ax1.plot(angles, ft_vals, color='#da5b3b', linewidth=2)
ax1.set_xticks(angles[:-1])
ax1.set_xticklabels(metrics, fontsize=11)
ax1.set_ylim(0, 1)
ax1.set_title("能力雷达图", fontsize=14, fontweight='bold', pad=20)
ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# ---- 图2: 逐题对比柱状图 ----
ax2 = axes[1]
x = np.arange(len(base_r))
width = 0.35
base_scores = [r["semantic_score"] for r in base_r]
ft_scores = [r["semantic_score"] for r in ft_r]
bars1 = ax2.bar(x - width/2, base_scores, width, label='原始模型', color='#4a6cf7', alpha=0.8)
bars2 = ax2.bar(x + width/2, ft_scores, width, label='微调模型', color='#da5b3b', alpha=0.8)
ax2.set_xlabel('题目编号', fontsize=11)
ax2.set_ylabel('语义相似度', fontsize=11)
ax2.set_title('逐题语义相似度对比', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([r["qid"] for r in base_r])
ax2.legend(fontsize=10)
ax2.set_ylim(0, 1)
ax2.grid(axis='y', alpha=0.3)

# ---- 图3: 指标汇总柱状图 ----
ax3 = axes[2]
labels = ["关键词命中", "语义相似度", "LLM-Judge"]
x3 = np.arange(len(labels))
base_avgs = [
    np.mean([r["keyword_score"] for r in base_r]),
    np.mean([r["semantic_score"] for r in base_r]),
    np.mean([float(r["judge_score"]) for r in base_r]) / 5,
]
ft_avgs = [
    np.mean([r["keyword_score"] for r in ft_r]),
    np.mean([r["semantic_score"] for r in ft_r]),
    np.mean([float(r["judge_score"]) for r in ft_r]) / 5,
]
bar3a = ax3.bar(x3 - 0.2, base_avgs, 0.35, label='原始模型', color='#4a6cf7', alpha=0.8)
bar3b = ax3.bar(x3 + 0.2, ft_avgs, 0.35, label='微调模型', color='#da5b3b', alpha=0.8)

# 数值标注
for bar, val in zip(bar3a, base_avgs):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}',
             ha='center', fontsize=9)
for bar, val in zip(bar3b, ft_avgs):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}',
             ha='center', fontsize=9)

ax3.set_xticks(x3)
ax3.set_xticklabels(labels, fontsize=11)
ax3.set_ylim(0, 1.1)
ax3.set_title('三维指标汇总', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("eval_report.png", dpi=150, bbox_inches='tight')
print("可视化报告已保存: eval_report.png")
