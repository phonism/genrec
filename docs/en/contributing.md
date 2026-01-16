# Contributing Guide

We welcome contributions to genrec! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.8 or higher
- PyTorch >= 1.11.0
- Git

### Installation

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/genrec.git
   cd genrec
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install in development mode:
   ```bash
   pip install -e .[dev]
   ```

## Development Workflow

### Code Style

We follow PEP 8 style guidelines. Please run the following before submitting:

```bash
# Format code
black genrec/ tests/
isort genrec/ tests/

# Check style
flake8 genrec/ tests/
mypy genrec/
```

### Testing

Run tests before submitting your changes:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=genrec

# Run specific test
pytest tests/test_datasets.py::test_p5_amazon_dataset
```

### Documentation

Update documentation when adding new features:

```bash
# Build documentation locally
cd docs
mkdocs serve
```

## Contributing Guidelines

### Issues

- Search existing issues before creating new ones
- Use clear, descriptive titles
- Provide steps to reproduce for bugs
- Include system information (OS, Python version, etc.)

### Pull Requests

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit:
   ```bash
   git commit -m "Add: brief description of changes"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Create a Pull Request on GitHub

### Commit Message Format

Use clear, descriptive commit messages:

- **Add**: New features or functionality
- **Fix**: Bug fixes
- **Update**: Changes to existing functionality
- **Docs**: Documentation changes
- **Test**: Adding or updating tests
- **Refactor**: Code refactoring without functional changes

Examples:
```
Add: TIGER model with transformer architecture
Fix: P5Amazon dataset loading for large categories
Update: configuration system to use dataclasses
Docs: API reference for dataset factory
Test: unit tests for text processors
```

## Types of Contributions

### Bug Fixes

- Fix issues reported in GitHub Issues
- Include test cases that reproduce the bug
- Update documentation if needed

### New Features

Before implementing major features:
1. Create an issue to discuss the feature
2. Get feedback from maintainers
3. Follow the existing architecture patterns

### Documentation

- Fix typos and improve clarity
- Add examples and tutorials
- Translate documentation (Chinese/English)
- Improve API documentation

### Performance Improvements

- Profile code to identify bottlenecks
- Include benchmarks showing improvements
- Ensure changes don't break existing functionality

## Code Architecture

### Adding New Datasets

To add a new dataset, follow these steps:

1. **Create the base dataset class**:
   ```python
   from genrec.data.base_dataset import BaseRecommenderDataset
   
   class MyDataset(BaseRecommenderDataset):
       def download(self):
           # Implement download logic
           pass
           
       def load_raw_data(self):
           # Implement data loading
           pass
           
       def preprocess_data(self, raw_data):
           # Implement preprocessing
           pass
   ```

2. **Create wrapper classes**:
   ```python
   from genrec.data.base_dataset import ItemDataset, SequenceDataset
   
   class MyItemDataset(ItemDataset):
       def __init__(self, **kwargs):
           # Initialize with your dataset
           pass
   
   class MySequenceDataset(SequenceDataset):
       def __init__(self, **kwargs):
           # Initialize with your dataset
           pass
   ```

3. **Add configuration**:
   ```python
   from genrec.data.configs import DatasetConfig
   
   @dataclass
   class MyDatasetConfig(DatasetConfig):
       # Add dataset-specific parameters
       special_param: str = "default_value"
   ```

4. **Register the dataset**:
   ```python
   from genrec.data.dataset_factory import DatasetFactory
   
   DatasetFactory.register_dataset(
       name="my_dataset",
       base_class=MyDataset,
       item_class=MyItemDataset,
       sequence_class=MySequenceDataset
   )
   ```

5. **Add tests and documentation**

For more details, please refer to the [API Documentation](api/datasets.md).

### Adding New Models

1. **Inherit from base classes**:
   ```python
   import torch.nn as nn
   
   class MyModel(nn.Module):
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
           
           # Define layers
           self.encoder = nn.Linear(input_dim, hidden_dim)
           self.decoder = nn.Linear(hidden_dim, output_dim)
           self.dropout = nn.Dropout(dropout)
       
       def forward(self, x):
           # Implement forward pass
           hidden = self.dropout(torch.relu(self.encoder(x)))
           output = self.decoder(hidden)
           return output
   ```

2. **Add to Gin configuration system**:
   ```python
   import gin
   
   @gin.configurable
   class MyModel(nn.Module):
       # Implementation
   ```

3. **Create training utilities**:
   ```python
   from genrec.trainers.base_trainer import BaseTrainer
   
   class MyModelTrainer(BaseTrainer):
       def __init__(self, model, config):
           super().__init__(model, config)
       
       def training_step(self, batch, batch_idx):
           # Implement training logic
           pass
   ```

4. **Add comprehensive tests**
5. **Update documentation**

## Testing Guidelines

### Unit Tests

- Test individual functions and classes
- Use pytest fixtures for setup
- Mock external dependencies
- Aim for >90% code coverage

```python
import pytest
from genrec.data import P5AmazonDataset

def test_p5_amazon_dataset_creation():
    config = P5AmazonConfig(
        root_dir="test_data",
        category="beauty"
    )
    dataset = P5AmazonDataset(config)
    assert dataset.category == "beauty"
```

### Integration Tests

- Test component interactions
- Use sample datasets
- Test end-to-end workflows

```python
def test_full_training_pipeline():
    # Test complete training workflow
    pass
```

### Performance Tests

- Benchmark critical operations
- Test with realistic data sizes
- Monitor memory usage

## Documentation Standards

### Docstring Format

Use Google-style docstrings:

```python
def process_data(data: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
    """Process input data with optional normalization.
    
    Args:
        data: Input DataFrame to process
        normalize: Whether to normalize numerical features
        
    Returns:
        Processed DataFrame
        
    Raises:
        ValueError: If data is empty
        
    Example:
        >>> df = pd.DataFrame({'col1': [1, 2, 3]})
        >>> result = process_data(df, normalize=True)
    """
```

### API Documentation

- Document all public methods and classes
- Include usage examples
- Explain parameters and return values
- Add type hints

### Tutorials and Guides

- Provide step-by-step instructions
- Include complete working examples
- Explain the reasoning behind design decisions
- Keep examples up-to-date with API changes

## Release Process

### Version Numbering

We follow Semantic Versioning (SemVer):
- MAJOR: Breaking changes
- MINOR: New features, backward compatible
- PATCH: Bug fixes, backward compatible

### Changelog

Update CHANGELOG.md with:
- New features
- Bug fixes
- Breaking changes
- Deprecations

## Getting Help

- Join our discussions on GitHub
- Ask questions in Issues
- Check existing documentation
- Review code examples

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help maintain a welcoming community
- Follow GitHub's Community Guidelines

Thank you for contributing to genrec!