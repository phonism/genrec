# 贡献指南

感谢您对 genrec 项目的关注！我们欢迎社区的贡献，无论是代码、文档、测试还是反馈。

## 开始贡献

### 环境设置

1. **Fork 项目**
   ```bash
   # 在 GitHub 上 fork 项目到您的账户
   # 然后克隆您的 fork
   git clone https://github.com/YOUR_USERNAME/genrec.git
   cd genrec
   ```

2. **设置开发环境**
   ```bash
   # 创建虚拟环境
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或 venv\Scripts\activate  # Windows
   
   # 安装开发依赖
   pip install -e ".[dev]"
   ```

3. **安装预提交钩子**
   ```bash
   pre-commit install
   ```

### 开发工作流

1. **创建功能分支**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **进行开发**
   - 遵循代码规范
   - 添加必要的测试
   - 更新相关文档

3. **提交更改**
   ```bash
   git add .
   git commit -m "Add: 简洁描述您的更改"
   ```

4. **推送到您的 fork**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **创建 Pull Request**
   - 在 GitHub 上创建 PR
   - 填写详细的 PR 描述
   - 等待代码审查

## 贡献类型

### 🐛 Bug 修复

发现了 bug？请：
1. 查看现有 issues 是否已报告
2. 如果没有，创建新的 issue 描述问题
3. 提供复现步骤和环境信息
4. 如果能修复，提交 PR

**Bug 报告模板：**
```markdown
## Bug 描述
简要描述 bug 的表现

## 复现步骤
1. 执行步骤1
2. 执行步骤2
3. 观察到错误

## 预期行为
描述您期望的正确行为

## 环境信息
- Python 版本：
- PyTorch 版本：
- 操作系统：
- GPU 信息（如果相关）：
```

### ✨ 新功能

添加新功能前：
1. 创建功能请求 issue 讨论设计
2. 确保功能符合项目目标
3. 考虑向后兼容性
4. 添加完整的测试和文档

**功能请求模板：**
```markdown
## 功能描述
描述您希望添加的功能

## 使用场景
解释这个功能的使用场景和价值

## 设计提案
如果有具体的设计想法，请详细描述

## 替代方案
考虑过的其他解决方案
```

### 📚 文档改进

文档贡献包括：
- 修复错别字和语法错误
- 改进代码示例
- 添加教程和指南
- 翻译内容

### 🧪 测试

测试贡献包括：
- 添加单元测试
- 改进测试覆盖率
- 添加集成测试
- 性能测试

## 代码规范

### Python 代码风格

我们使用以下工具保持代码一致性：

- **Black**: 代码格式化
- **isort**: 导入排序
- **flake8**: 代码检查
- **mypy**: 类型检查

配置文件在项目根目录：
- `pyproject.toml`
- `.flake8`
- `mypy.ini`

### 代码风格示例

```python
from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
from genrec.base import BaseModel


class ExampleModel(BaseModel):
    """示例模型类.
    
    Args:
        input_dim: 输入维度
        hidden_dim: 隐藏层维度
        output_dim: 输出维度
        dropout: Dropout 概率
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 定义层
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播.
        
        Args:
            x: 输入张量 (batch_size, input_dim)
            
        Returns:
            输出张量 (batch_size, output_dim)
        """
        h = self.linear1(x)
        h = torch.relu(h)
        h = self.dropout(h)
        output = self.linear2(h)
        
        return output
```

### 提交信息格式

使用清晰的提交信息：

```
类型: 简短描述 (不超过 50 字符)

详细描述（如果需要）：
- 解释为什么做这个更改
- 提及任何重要的技术细节
- 引用相关的 issue (#123)

Breaking change: 如果有破坏性更改，说明影响
```

**提交类型：**
- `Add`: 新功能
- `Fix`: Bug 修复
- `Update`: 更新现有功能
- `Remove`: 删除功能
- `Refactor`: 重构代码
- `Docs`: 文档更改
- `Test`: 测试相关
- `Style`: 代码风格调整

## 测试指南

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_models.py

# 运行特定测试
pytest tests/test_models.py::test_rqvae_forward

# 生成覆盖率报告
pytest --cov=genrec
```

### 编写测试

测试文件结构：
```
tests/
├── conftest.py                 # 测试配置和固件
├── test_models/
│   ├── test_rqvae.py          # RQVAE 测试
│   └── test_tiger.py          # TIGER 测试
├── test_data/
│   ├── test_datasets.py       # 数据集测试
│   └── test_processors.py     # 处理器测试
└── test_utils/
    └── test_metrics.py         # 工具函数测试
```

测试示例：
```python
import pytest
import torch
from genrec.models.rqvae import RqVae


class TestRqVae:
    """RQVAE 模型测试"""
    
    @pytest.fixture
    def model(self):
        """创建测试模型"""
        return RqVae(
            input_dim=768,
            hidden_dim=256,
            latent_dim=128,
            num_embeddings=512
        )
    
    @pytest.fixture
    def sample_input(self):
        """创建样本输入"""
        return torch.randn(32, 768)
    
    def test_forward(self, model, sample_input):
        """测试前向传播"""
        reconstructed, commitment_loss, embedding_loss, sem_ids = model(sample_input)
        
        # 检查输出形状
        assert reconstructed.shape == sample_input.shape
        assert sem_ids.shape == (32,)
        
        # 检查损失类型
        assert isinstance(commitment_loss, torch.Tensor)
        assert isinstance(embedding_loss, torch.Tensor)
    
    def test_generate_semantic_ids(self, model, sample_input):
        """测试语义ID生成"""
        sem_ids = model.generate_semantic_ids(sample_input)
        
        assert sem_ids.shape == (32,)
        assert sem_ids.dtype == torch.long
        assert torch.all(sem_ids >= 0)
        assert torch.all(sem_ids < model.quantizer.num_embeddings)
    
    @pytest.mark.parametrize("input_dim,hidden_dim", [
        (256, 128),
        (512, 256),
        (1024, 512),
    ])
    def test_different_dimensions(self, input_dim, hidden_dim):
        """测试不同维度配置"""
        model = RqVae(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=64,
            num_embeddings=256
        )
        
        sample_input = torch.randn(16, input_dim)
        output = model(sample_input)
        
        assert output[0].shape == (16, input_dim)
```

## 文档贡献

### 文档结构

```
docs/
├── zh/                         # 中文文档
│   ├── index.md               # 首页
│   ├── installation.md        # 安装指南
│   ├── quickstart.md          # 快速开始
│   ├── models/                # 模型文档
│   ├── dataset/               # 数据集文档
│   ├── training/              # 训练指南
│   └── api/                   # API 参考
└── en/                        # 英文文档
    └── ...                    # 同样的结构
```

### 编写文档

1. **使用 Markdown 格式**
2. **包含代码示例**
3. **添加适当的链接**
4. **保持简洁明了**

文档示例：
```markdown
# 模型训练

本指南介绍如何训练 genrec 模型。

## 快速开始

最简单的训练方式：

```python
from genrec.models.rqvae import RqVae
from genrec.data.p5_amazon import P5AmazonItemDataset

# 加载数据
dataset = P5AmazonItemDataset(root="data", split="beauty")

# 创建模型
model = RqVae(input_dim=768, num_embeddings=1024)

# 训练（详细代码见下文）
trainer.fit(model, dataloader)
```

## 详细配置

### 数据配置

| 参数 | 类型 | 默认值 | 描述 |
|------|------|---------|------|
| `root` | str | - | 数据集根目录 |
| `split` | str | "beauty" | 数据分割 |

更多详细信息请参考 [API 文档](api/datasets.md)。
```

### 本地预览

```bash
# 安装 MkDocs
pip install mkdocs mkdocs-material

# 启动本地服务器
mkdocs serve

# 在浏览器中访问 http://127.0.0.1:8000
```

## 发布流程

### 版本管理

我们使用 [语义化版本控制](https://semver.org/lang/zh-CN/)：

- **MAJOR**: 不兼容的 API 更改
- **MINOR**: 向后兼容的功能添加
- **PATCH**: 向后兼容的错误修复

### 发布检查清单

发布新版本前：

- [ ] 所有测试通过
- [ ] 文档已更新
- [ ] CHANGELOG.md 已更新
- [ ] 版本号已更新
- [ ] 创建 Git 标签

## 代码审查

### 审查标准

代码审查关注：

1. **正确性**: 代码是否解决了问题
2. **可读性**: 代码是否易于理解
3. **可维护性**: 代码是否易于修改
4. **性能**: 是否有明显的性能问题
5. **测试**: 是否有充分的测试覆盖

### 审查流程

1. **自动检查**: CI/CD 管道运行测试
2. **代码审查**: 维护者审查代码
3. **反馈和修改**: 根据反馈进行调整
4. **批准和合并**: 审查通过后合并

## 获得帮助

如果您在贡献过程中遇到问题：

1. **查看现有文档和 FAQ**
2. **搜索现有 issues**
3. **在 GitHub Discussions 中提问**
4. **创建新的 issue**

## 行为准则

我们致力于为每个人提供友好、安全和欢迎的环境。请：

- **保持尊重**: 尊重不同的观点和经验
- **保持包容**: 欢迎新手和不同背景的贡献者
- **保持建设性**: 提供有帮助的反馈和建议
- **保持专业**: 专注于技术讨论

## 联系方式

- **GitHub Issues**: 技术问题和 Bug 报告
- **GitHub Discussions**: 一般讨论和问答
- **邮件**: [项目邮箱]（如果有）

感谢您对 genrec 项目的贡献！🎉