# RQVAE 训练

本指南详细介绍如何训练 RQVAE 模型。

## 训练准备

### 1. 数据准备

确保数据集已经下载并放置在正确位置：

```bash
# 数据将自动下载到指定目录
mkdir -p dataset/amazon
```

### 2. 检查配置文件

查看默认配置：

```bash
cat config/rqvae/p5_amazon.gin
```

主要配置参数：

```gin
# 训练参数
train.iterations=400000          # 训练迭代数
train.learning_rate=0.0005      # 学习率
train.batch_size=64             # 批量大小
train.weight_decay=0.01         # 权重衰减

# 模型参数
train.vae_input_dim=768         # 输入维度
train.vae_embed_dim=32          # 嵌入维度
train.vae_hidden_dims=[512, 256, 128]  # 隐藏层维度
train.vae_codebook_size=256     # 码本大小
train.vae_n_layers=3            # 量化层数

# 量化设置
train.vae_codebook_mode=%genrec.models.rqvae.QuantizeForwardMode.ROTATION_TRICK
train.commitment_weight=0.25    # 承诺损失权重
```

## 开始训练

### 基本训练命令

```bash
python genrec/trainers/rqvae_trainer.py config/rqvae/p5_amazon.gin
```

### 训练过程监控

如果启用了 Weights & Biases：

```gin
train.wandb_logging=True
train.wandb_project="my_rqvae_project"
```

### GPU 训练

使用多GPU训练：

```bash
accelerate config  # 首次运行时配置
accelerate launch genrec/trainers/rqvae_trainer.py config/rqvae/p5_amazon.gin
```

## 自定义配置

### 创建自定义配置文件

```gin
# my_rqvae_config.gin
import genrec.data.p5_amazon
import genrec.models.rqvae

# 自定义训练参数
train.iterations=200000
train.batch_size=32
train.learning_rate=0.001

# 自定义模型架构
train.vae_embed_dim=64
train.vae_hidden_dims=[512, 256, 128, 64]
train.vae_codebook_size=512

# 数据路径
train.dataset_folder="path/to/my/dataset"
train.save_dir_root="path/to/my/output"

# 实验跟踪
train.wandb_logging=True
train.wandb_project="custom_rqvae_experiment"
```

使用自定义配置：

```bash
python genrec/trainers/rqvae_trainer.py my_rqvae_config.gin
```

## 训练监控

### 关键指标

训练过程中关注以下指标：

- **总损失 (Total Loss)**: 整体训练损失
- **重构损失 (Reconstruction Loss)**: 重构质量
- **量化损失 (Quantization Loss)**: 量化效果
- **承诺损失 (Commitment Loss)**: 编码器承诺度

### 日志输出示例

```
Epoch 1000: Loss=2.3456, Recon=2.1234, Quant=0.1234, Commit=0.0988
Epoch 2000: Loss=1.9876, Recon=1.8234, Quant=0.0987, Commit=0.0655
...
```

## 模型评估

### 重构质量评估

```python
from genrec.models.rqvae import RqVae
from genrec.data.p5_amazon import P5AmazonItemDataset

# 加载训练好的模型
model = RqVae.load_from_checkpoint("out/rqvae/checkpoint_299999.pt")

# 评估数据集
eval_dataset = P5AmazonItemDataset(
    root="dataset/amazon",
    train_test_split="eval"
)

# 计算重构损失
model.eval()
with torch.no_grad():
    eval_loss = model.evaluate(eval_dataset)
    print(f"Evaluation loss: {eval_loss:.4f}")
```

### 码本利用率分析

```python
def analyze_codebook_usage(model, dataloader):
    used_codes = set()
    
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(batch)
            semantic_ids = outputs.sem_ids
            used_codes.update(semantic_ids.flatten().tolist())
    
    usage_rate = len(used_codes) / model.codebook_size
    print(f"Codebook usage: {usage_rate:.2%}")
    print(f"Used codes: {len(used_codes)}/{model.codebook_size}")
    
    return used_codes
```

## 故障排除

### 常见问题

**Q: 训练损失不收敛？**

A: 尝试以下解决方案：
- 降低学习率：`train.learning_rate=0.0001`
- 调整承诺权重：`train.commitment_weight=0.1`
- 检查数据预处理是否正确

**Q: 码本崩塌（所有样本使用同一个码）？**

A: 
- 使用 ROTATION_TRICK 模式
- 增加承诺权重
- 减小学习率

**Q: GPU 内存不足？**

A:
- 减小批量大小：`train.batch_size=32`
- 减小模型规模：`train.vae_hidden_dims=[256, 128]`
- 启用混合精度训练

### 调试技巧

1. **梯度检查**：
```python
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"{name}: {grad_norm:.6f}")
```

2. **损失分析**：
```python
# 分别打印各个损失组件
print(f"Reconstruction: {outputs.reconstruction_loss:.4f}")
print(f"Quantization: {outputs.quantization_loss:.4f}")
print(f"Commitment: {outputs.commitment_loss:.4f}")
```

## 最佳实践

### 超参数调优建议

1. **学习率调度**：
```gin
# 使用余弦退火
train.scheduler="cosine"
train.min_lr=1e-6
```

2. **早停策略**：
```gin
train.early_stopping=True
train.patience=10000
```

3. **模型保存频率**：
```gin
train.save_model_every=50000  # 每5万次迭代保存一次
train.eval_every=10000        # 每1万次迭代评估一次
```

### 实验管理

建议使用版本控制和实验跟踪：

```bash
# 创建实验分支
git checkout -b experiment/rqvae-large-codebook

# 修改配置
vim config/rqvae/large_codebook.gin

# 运行实验
python genrec/trainers/rqvae_trainer.py config/rqvae/large_codebook.gin

# 记录结果
git add .
git commit -m "Experiment: large codebook (size=1024)"
```

## 下一步

训练完成后，你可以：

1. 使用训练好的 RQVAE 进行 [TIGER 训练](tiger.md)
2. 分析[模型性能](../models/rqvae.md#评估指标)
3. 尝试[不同的数据集](../dataset/custom.md)