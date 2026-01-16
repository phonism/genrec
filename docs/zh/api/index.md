# API 参考

本节提供 genrec 框架的详细 API 文档。

## 核心模块

### 模型

- **[RQVAE](rqvae.md)**: 残差量化变分自编码器
- **[TIGER](tiger.md)**: 基于 Transformer 的生成式检索模型

### 数据处理

- **[基础数据集](base-dataset.md)**: 数据集抽象基类
- **[配置管理](configs.md)**: 配置管理类
- **[处理器](processors.md)**: 文本和序列处理工具
- **[数据集工厂](dataset-factory.md)**: 数据集创建工厂模式

### 训练

- **[训练器](trainers.md)**: 训练工具和脚本
- **[模块](modules.md)**: 核心构建块（编码器、损失函数、指标）

## 快速导航

### 核心组件

**模型:**
- [RQVAE 模型类](../models/rqvae.md) - 向量量化变分自编码器
- [TIGER 模型类](../models/tiger.md) - 基于 Transformer 的生成式检索

**数据处理:**
- [数据集类](../dataset/overview.md) - 数据加载和预处理
- [配置系统](../dataset/overview.md) - 参数管理
- [处理器](../dataset/overview.md) - 文本和序列处理工具

**训练:**
- [RQVAE 训练](../training/rqvae.md) - 训练流程和配置
- [TIGER 训练](../training/tiger.md) - 高级训练工作流

### 代码示例

查看[示例](../examples.md)页面了解实用的使用模式和代码片段。