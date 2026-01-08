# GenRec

A PyTorch benchmark for generative recommendation systems, including RQVAE, TIGER, and COBRA.

## Benchmark Results

**Validation Set Metrics**

| Methods | | Sports | | | | Beauty | | | | Toys | | | |
|---------|--|--------|--|--|--|--------|--|--|--|------|--|--|--|
| | R@5 | N@5 | R@10 | N@10 | R@5 | N@5 | R@10 | N@10 | R@5 | N@5 | R@10 | N@10 |
| TIGER (Paper) | 0.0264 | 0.0181 | 0.0400 | 0.0225 | 0.0454 | 0.0321 | 0.0648 | 0.0384 | 0.0521 | 0.0371 | 0.0712 | 0.0432 |
| TIGER (Ours) | 0.0266 | 0.0176 | 0.0414 | 0.0224 | 0.0465 | 0.0297 | 0.0721 | 0.0378 | 0.0420 | 0.0280 | 0.0647 | 0.0350 |

**Experiment Tracking:** [Beauty](https://wandb.ai/luckyqueen/amazon_beauty_tiger_training) | [Sports](https://wandb.ai/luckyqueen/amazon_sports_tiger_training) | [Toys](https://wandb.ai/luckyqueen/amazon_toys_tiger_training/)

## Installation

```bash
pip install -r requirements.txt
```

## Supported Datasets

- Amazon Beauty
- Amazon Sports
- Amazon Toys

Data will be automatically downloaded on first run.

## RQVAE Training

RQVAE generates semantic IDs for items using residual quantization.

```bash
# Train on Beauty (default)
python genrec/trainers/rqvae_trainer.py config/rqvae/amazon.gin

# Train on other datasets
python genrec/trainers/rqvae_trainer.py config/rqvae/amazon.gin --split sports
python genrec/trainers/rqvae_trainer.py config/rqvae/amazon.gin --split toys
```

## TIGER Training

TIGER is a generative retrieval model for sequential recommendation.

```bash
# Train on Beauty (default)
python genrec/trainers/tiger_trainer.py config/tiger/amazon.gin

# Train on other datasets
python genrec/trainers/tiger_trainer.py config/tiger/amazon.gin --split sports
python genrec/trainers/tiger_trainer.py config/tiger/amazon.gin --split toys
```

**Note:** TIGER requires a pretrained RQVAE checkpoint. Train RQVAE first.

## Command Line Options

### Dataset Selection
```bash
--split <dataset>  # beauty, sports, toys
```

### Gin Parameter Override
```bash
--gin "param=value"  # Override any config parameter
```

### Examples
```bash
# Change epochs and batch size
python trainer.py config.gin --gin "train.epochs=200" --gin "train.batch_size=128"

# Set wandb run name
python trainer.py config.gin --gin 'train.wandb_run_name="my_experiment"'
```

## References

- [TIGER](https://arxiv.org/abs/2305.05065): Recommender Systems with Generative Retrieval
- [RQ-VAE-Recommender](https://github.com/EdoardoBotta/RQ-VAE-Recommender) by Edoardo Botta
