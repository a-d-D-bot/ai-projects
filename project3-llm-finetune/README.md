# 🧠 中文大模型指令微调

基于 Qwen2.5-1.5B-Instruct + QLoRA 的指令微调项目。

## 技术栈

`Qwen2.5` · `LoRA/QLoRA` · `LLaMA-Factory` · `bitsandbytes`

## 项目结构

```
├── data/my_qa.json       # 40条中文知识问答数据集
├── configs/sft.yaml      # QLoRA 微调配置
├── compare.py            # 微调前后效果对比脚本
└── README.md
```

## 快速复现

```bash
# 1. 克隆 LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"

# 2. 下载模型（国内推荐 ModelScope）
python -c "from modelscope import snapshot_download; snapshot_download('qwen/Qwen2.5-1.5B-Instruct', cache_dir='./models')"

# 3. 复制数据 + 注册
cp ../data/my_qa.json data/
# 在 data/dataset_info.json 中注册 "my_qa"

# 4. 开始训练
llamafactory-cli train ../configs/sft.yaml

# 5. 对比效果
python ../compare.py
```

## 训练结果

| 指标 | 值 |
|---|---|
| 模型 | Qwen2.5-1.5B-Instruct |
| 微调方法 | QLoRA (rank=8, 4bit) |
| 数据量 | 40 条 |
| 训练时长 | 53 秒（RTX 4060 8GB） |
| 最终 Loss | 1.79 |
