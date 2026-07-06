"""
大模型评测框架
三种打分: ①关键词匹配 ②语义相似度 ③LLM-as-Judge
对比: 原始 Qwen2.5 vs 微调后 Qwen2.5
"""
import json
import sys
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# LLaMA-Factory 的路径
sys.path.insert(0, r"C:\Users\颜灿\Desktop\实习相关\ai-projects\project3-llm-finetune\LLaMA-Factory\src")
from llamafactory.chat import ChatModel
import torch

# ============================================================
# 0. 加载评测数据集
# ============================================================
with open("eval_dataset.json", "r", encoding="utf-8") as f:
    eval_data = json.load(f)
print(f"加载评测集: {len(eval_data)} 题\n")

# ============================================================
# 1. 加载语义模型（用于语义相似度打分）
# ============================================================
print("加载语义模型...")
sem_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("语义模型就绪\n")

# ============================================================
# 2. 加载两个待评测模型
# ============================================================
MODEL_PATH = r"C:\Users\颜灿\Desktop\实习相关\ai-projects\project3-llm-finetune\LLaMA-Factory\models\qwen\Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = r"C:\Users\颜灿\Desktop\实习相关\ai-projects\project3-llm-finetune\LLaMA-Factory\saves\qwen25-1.5b\qlora\sft"

print("加载原始模型...")
base_model = ChatModel({
    "model_name_or_path": MODEL_PATH,
    "template": "qwen",
    "infer_backend": "huggingface",
    "trust_remote_code": True,
})

print("加载微调模型...")
finetuned_model = ChatModel({
    "model_name_or_path": MODEL_PATH,
    "adapter_name_or_path": ADAPTER_PATH,
    "template": "qwen",
    "infer_backend": "huggingface",
    "trust_remote_code": True,
})
print("两个模型就绪\n")


# ============================================================
# 3. 打分函数
# ============================================================
def keyword_score(answer, keywords):
    """关键词命中率"""
    if not answer:
        return 0.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords)

def semantic_score(answer, reference):
    """语义相似度"""
    if not answer or not reference:
        return 0.0
    emb = sem_model.encode([answer, reference])
    return float(cosine_similarity([emb[0]], [emb[1]])[0][0])

def llm_judge_score(question, reference, answer, judge_model):
    """LLM-as-Judge 打分 (1-5)"""
    if not answer:
        return 0.0, "无回答"

    prompt = f"""你是一个专业的AI模型评测裁判。请根据以下标准对模型回答打分（1-5分，1分最差，5分最好）：

【评分维度】
- 准确性(40%): 回答内容是否正确，与标准答案的事实是否一致
- 完整性(30%): 是否覆盖了标准答案的核心要点
- 流畅性(20%): 语言表达是否通顺自然
- 安全性(10%): 回答是否安全无害

【问题】
{question}

【标准答案】
{reference}

【模型回答】
{answer}

请严格按以下JSON格式输出（不要输出其他内容）：
{{"准确性": X, "完整性": X, "流畅性": X, "安全性": X, "总分": "X.X", "评语": "一句话评价"}}"""

    try:
        resp = judge_model.chat([{"role": "user", "content": prompt}])[0].response_text
        # 尝试解析 JSON
        resp = resp.strip()
        if "```" in resp:
            resp = resp.split("```")[1]
            if resp.startswith("json"):
                resp = resp[4:]
        result = json.loads(resp)
        judge_total = result.get("总分", 0)
        return float(judge_total), result.get("评语", "")
    except Exception as e:
        return 0.0, f"解析失败: {str(e)[:40]}"


# ============================================================
# 4. 跑评测
# ============================================================
results = []

for i, item in enumerate(eval_data):
    qid = item["id"]
    question = item["question"]
    reference = item["reference"]
    keywords = item["keywords"]

    print(f"[{i+1}/{len(eval_data)}] Q{qid}: {question[:40]}...")

    # 两个模型分别回答
    base_ans = base_model.chat([{"role": "user", "content": question}])[0].response_text
    ft_ans = finetuned_model.chat([{"role": "user", "content": question}])[0].response_text

    torch.cuda.empty_cache()

    # 对两个模型的回答分别打分
    for model_name, answer in [("原始模型", base_ans), ("微调模型", ft_ans)]:
        kw = keyword_score(answer, keywords)
        sem = semantic_score(answer, reference)
        judge_total, judge_comment = llm_judge_score(question, reference, answer, finetuned_model)

        results.append({
            "qid": qid,
            "question": question[:60],
            "model": model_name,
            "answer": answer[:200],
            "keyword_score": round(kw, 3),
            "semantic_score": round(sem, 3),
            "judge_score": judge_total,
            "judge_comment": judge_comment,
        })

    # 打印当前结果
    base_row = results[-2]
    ft_row = results[-1]
    print(f"  原始: KW={base_row['keyword_score']:.2f} SEM={base_row['semantic_score']:.2f} JUDGE={base_row['judge_score']}")
    print(f"  微调: KW={ft_row['keyword_score']:.2f} SEM={ft_row['semantic_score']:.2f} JUDGE={ft_row['judge_score']}")

print("\n评测完成！")

# ============================================================
# 5. 汇总统计
# ============================================================
base_r = [r for r in results if r["model"] == "原始模型"]
ft_r = [r for r in results if r["model"] == "微调模型"]

print("\n" + "=" * 50)
print("📊 汇总对比")
print("=" * 50)
for metric, name in [("keyword_score", "关键词命中"), ("semantic_score", "语义相似度"), ("judge_score", "LLM-Judge")]:
    base_avg = np.mean([r[metric] for r in base_r])
    ft_avg = np.mean([r[metric] for r in ft_r])
    diff = ft_avg - base_avg
    arrow = "↑" if diff > 0 else "↓" if diff < 0 else "→"
    print(f"{name:　<8}: 原始 {base_avg:.3f} | 微调 {ft_avg:.3f} | {arrow} {abs(diff):.3f}")

# 保存结果
with open("eval_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\n结果已保存到 eval_results.json")
