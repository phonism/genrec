# 常见问题 (FAQ)

本页面收集了使用 genrec 框架时的常见问题和解答。

## 安装和环境

### Q: 如何安装 genrec？

A: 目前支持从源代码安装：

```bash
git clone https://github.com/phonism/genrec.git
cd genrec
pip install -e .
```

### Q: 支持哪些 Python 版本？

A: 推荐使用 Python 3.8 或更高版本。框架在 Python 3.8、3.9、3.10 上经过测试。

### Q: 需要哪些主要依赖？

A: 主要依赖包括：
- PyTorch >= 1.11.0
- PyTorch Lightning >= 1.6.0
- sentence-transformers >= 2.2.0
- pandas >= 1.3.0
- numpy >= 1.21.0

### Q: 是否支持 GPU 训练？

A: 是的，框架完全支持 GPU 训练。确保安装了正确的 PyTorch CUDA 版本。

## 数据和数据集

### Q: 支持哪些数据集格式？

A: 框架支持：
- JSON 格式的推荐数据集
- CSV 格式的用户-物品交互数据
- Parquet 格式的大规模数据集
- 自定义格式（通过继承基类实现）

### Q: 如何添加自定义数据集？

A: 继承 `BaseRecommenderDataset` 类并实现必要方法：

```python
from genrec.data.base_dataset import BaseRecommenderDataset

class MyDataset(BaseRecommenderDataset):
    def download(self):
        # 实现数据下载逻辑
        pass
        
    def load_raw_data(self):
        # 实现数据加载逻辑
        return {"items": items_df, "interactions": interactions_df}
        
    def preprocess_data(self, raw_data):
        # 实现数据预处理逻辑
        return processed_data
```

### Q: P5 Amazon 数据集有多大？

A: 不同类别的大小不同：
- Beauty: ~500MB
- Electronics: ~2GB
- Sports: ~1GB
- 完整数据集可能需要 10GB+ 的存储空间

### Q: 如何处理缺失的物品特征？

A: 框架自动处理缺失特征：
- 文本字段用 "Unknown" 填充
- 数值字段用均值或 0 填充
- 可以在配置中自定义填充策略

## 模型训练

### Q: RQVAE 训练需要多长时间？

A: 取决于数据集大小和硬件：
- 小数据集（<10万物品）：30分钟 - 2小时
- 中等数据集（10-100万物品）：2-8小时
- 大数据集（>100万物品）：8-24小时

### Q: TIGER 训练的内存要求是什么？

A: 典型内存使用：
- 最小：8GB GPU 内存（小批量大小）
- 推荐：16GB GPU 内存（中等批量大小）
- 大规模：32GB+ GPU 内存（大批量大小）

### Q: 如何选择合适的嵌入维度？

A: 经验法则：
- 小数据集：256-512 维
- 中等数据集：512-768 维
- 大数据集：768-1024 维
- 具体选择应基于验证集性能

### Q: 训练过程中出现 CUDA 内存不足怎么办？

A: 解决方法：
1. 减少批量大小
2. 使用梯度累积
3. 启用混合精度训练
4. 减少模型尺寸

```python
# 减少批量大小
config.batch_size = 16

# 梯度累积
config.accumulate_grad_batches = 4

# 混合精度
config.precision = 16
```

## 模型使用

### Q: 如何生成推荐？

A: 基本推荐生成：

```python
# 加载模型
rqvae = RqVae.load_from_checkpoint("rqvae.ckpt")
tiger = Tiger.load_from_checkpoint("tiger.ckpt")

# 用户历史（物品ID）
user_history = [1, 5, 23, 67]

# 转换为语义ID
semantic_ids = rqvae.encode_items(user_history)

# 生成推荐
recommendations = tiger.generate(semantic_ids, max_length=10)
```

### Q: 如何处理冷启动用户？

A: 对于新用户：
1. 使用流行物品推荐
2. 基于用户画像的内容推荐
3. 基于物品特征的相似度推荐

```python
def recommend_for_new_user(user_profile, k=10):
    # 基于用户画像找相似物品
    similar_items = find_similar_items_by_profile(user_profile)
    return similar_items[:k]
```

### Q: 推荐结果的多样性如何保证？

A: 提高多样性的方法：
1. 使用 Top-p 采样而不是贪心采样
2. 后处理去重和多样化
3. 在训练时加入多样性损失

```python
# 使用采样生成
recommendations = tiger.generate(
    input_seq, 
    temperature=0.8,
    top_p=0.9,
    do_sample=True
)
```

## 性能和优化

### Q: 如何提高推理速度？

A: 优化方法：
1. 模型量化
2. ONNX 导出
3. TensorRT 优化
4. 批量推理

```python
# 模型量化
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# 批量推理
def batch_recommend(user_histories, batch_size=32):
    results = []
    for i in range(0, len(user_histories), batch_size):
        batch = user_histories[i:i+batch_size]
        batch_results = model.batch_generate(batch)
        results.extend(batch_results)
    return results
```

### Q: 模型大小可以压缩吗？

A: 压缩技术：
- 量化：减少 50-75% 大小
- 剪枝：移除不重要的参数
- 知识蒸馏：训练小模型模仿大模型

### Q: 如何监控模型性能？

A: 监控指标：
- 推理延迟
- 内存使用
- GPU 利用率
- 推荐质量指标（Recall、NDCG）

## 错误和调试

### Q: 训练时出现 "RuntimeError: CUDA out of memory" 错误？

A: 解决步骤：
1. 减少 batch_size
2. 启用梯度检查点
3. 清理 GPU 缓存

```python
import torch
torch.cuda.empty_cache()

# 或在训练配置中
config.gradient_checkpointing = True
```

### Q: 模型加载失败怎么办？

A: 检查项：
1. 检查点文件是否完整
2. PyTorch 版本兼容性
3. 模型架构是否匹配

```python
try:
    model = Model.load_from_checkpoint(checkpoint_path)
except Exception as e:
    print(f"加载失败: {e}")
    # 尝试加载状态字典
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
```

### Q: 训练损失不收敛怎么办？

A: 调试方法：
1. 检查学习率（可能过大或过小）
2. 检查数据预处理
3. 增加训练数据量
4. 调整模型架构

```python
# 学习率调度
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=5, factor=0.5
)
```

## 部署和生产

### Q: 如何部署到生产环境？

A: 部署选项：
1. REST API 服务（FastAPI/Flask）
2. Docker 容器化
3. Kubernetes 集群
4. 云服务平台

### Q: 支持实时推荐吗？

A: 是的，框架支持：
- 在线推理 API
- 批量预计算
- 流式处理集成

### Q: 如何进行 A/B 测试？

A: A/B 测试框架：

```python
class ABTestFramework:
    def get_variant(self, user_id, experiment_name):
        # 基于用户ID的一致性哈希分组
        hash_value = hash(f"{user_id}_{experiment_name}")
        return "A" if hash_value % 2 == 0 else "B"
    
    def recommend_with_test(self, user_id, user_history):
        variant = self.get_variant(user_id, "model_test")
        if variant == "A":
            return model_a.recommend(user_history)
        else:
            return model_b.recommend(user_history)
```

## 高级使用

### Q: 如何实现多任务学习？

A: 扩展模型以支持多个目标：

```python
class MultiTaskTiger(Tiger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rating_head = nn.Linear(self.embedding_dim, 1)
        self.category_head = nn.Linear(self.embedding_dim, num_categories)
    
    def forward(self, x):
        hidden = super().forward(x)
        
        # 多个输出头
        recommendations = self.recommendation_head(hidden)
        ratings = self.rating_head(hidden)
        categories = self.category_head(hidden)
        
        return recommendations, ratings, categories
```

### Q: 如何集成外部特征？

A: 特征融合方法：

```python
class FeatureEnhancedModel(Tiger):
    def __init__(self, user_feature_dim, item_feature_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user_feature_proj = nn.Linear(user_feature_dim, self.embedding_dim)
        self.item_feature_proj = nn.Linear(item_feature_dim, self.embedding_dim)
    
    def forward(self, item_seq, user_features=None, item_features=None):
        seq_emb = super().forward(item_seq)
        
        if user_features is not None:
            user_emb = self.user_feature_proj(user_features)
            seq_emb = seq_emb + user_emb.unsqueeze(1)
        
        return seq_emb
```

### Q: 如何处理序列中的时间信息？

A: 时间感知的推荐：

```python
class TimeAwareTiger(Tiger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_emb = nn.Embedding(24 * 7, self.embedding_dim)  # 小时*天
    
    def forward(self, item_seq, time_seq=None):
        seq_emb = self.item_embedding(item_seq)
        
        if time_seq is not None:
            time_emb = self.time_emb(time_seq)
            seq_emb = seq_emb + time_emb
        
        return self.transformer(seq_emb)
```

如果您有其他问题，请查阅 [API 文档](api/index.md) 或在 GitHub 上提交 issue。