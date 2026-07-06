"""微调前后效果对比"""
import sys
sys.path.insert(0, "src")

from llamafactory.chat import ChatModel
import torch

# 测试问题（选3个你数据里有的）
questions = [
    "什么是过拟合？如何防止？",
    "Python中列表和元组有什么区别？",
    "什么是LoRA以及它的核心思想？",
]

# ---- 1. 微调前（原始模型）----
print("=" * 60)
print("📌 微调前（原始 Qwen2.5-1.5B-Instruct）")
print("=" * 60)

base = ChatModel({
    "model_name_or_path": "./models/qwen/Qwen2.5-1.5B-Instruct",
    "template": "qwen",
    "infer_backend": "huggingface",
    "trust_remote_code": True,
})

for q in questions:
    resp = base.chat([{"role": "user", "content": q}])[0].response_text
    print(f"\nQ: {q}")
    print(f"A: {resp[:300]}...")
    print("-" * 40)

del base
torch.cuda.empty_cache()

# ---- 2. 微调后（加 LoRA adapter）----
print("\n\n" + "=" * 60)
print("📌 微调后（+ LoRA adapter）")
print("=" * 60)

finetuned = ChatModel({
    "model_name_or_path": "./models/qwen/Qwen2.5-1.5B-Instruct",
    "adapter_name_or_path": "saves/qwen25-1.5b/qlora/sft",
    "template": "qwen",
    "infer_backend": "huggingface",
    "trust_remote_code": True,
})

for q in questions:
    resp = finetuned.chat([{"role": "user", "content": q}])[0].response_text
    print(f"\nQ: {q}")
    print(f"A: {resp[:300]}...")
    print("-" * 40)

print("\n\n对比完成！")
