"""
检查 Reference TIGER 数据划分是否存在数据泄露
"""
import pandas as pd
import numpy as np

DATA_DIR = "/root/workspace/code/GenerativeRecommenders/reference/TIGER/data/Beauty"

train_df = pd.read_parquet(f'{DATA_DIR}/train.parquet')
valid_df = pd.read_parquet(f'{DATA_DIR}/valid.parquet')
test_df = pd.read_parquet(f'{DATA_DIR}/test.parquet')

print("=" * 60)
print("数据集大小")
print("=" * 60)
print(f"Train: {len(train_df)}")
print(f"Valid: {len(valid_df)}")
print(f"Test: {len(test_df)}")

print("\n" + "=" * 60)
print("用户1的数据示例")
print("=" * 60)
for name, df in [("Train", train_df), ("Valid", valid_df), ("Test", test_df)]:
    row = df[df['user'] == 1].iloc[0]
    print(f"{name}:")
    print(f"  history: {list(row['history'])}")
    print(f"  target: {row['target']}")

print("\n" + "=" * 60)
print("检查数据泄露")
print("=" * 60)

# 检查: valid的target是否会出现在train的滑动窗口中?
# train.parquet: history=[...], target=items[-2]
# 滑动窗口会生成: history=train_history+[train_target], target=?
# 这个target就是valid的target!

leakage_count = 0
for idx, row in valid_df.iterrows():
    user = row['user']
    valid_target = row['target']

    train_row = train_df[train_df['user'] == user].iloc[0]
    train_history = list(train_row['history'])
    train_target = train_row['target']

    # 滑动窗口会产生的序列
    # train_parquet存的sequence = train_history + [train_target]
    # 滑动窗口展开后，最后一个样本是:
    #   history = train_history + [train_target] = valid_history (去掉最后一个)
    #   target = valid_target

    # 实际检查: valid_history的最后一个是train_target吗?
    valid_history = list(row['history'])
    if len(valid_history) > 0 and valid_history[-1] == train_target:
        leakage_count += 1

print(f"Valid target 会在 Train 滑动窗口中出现的样本数: {leakage_count}/{len(valid_df)}")

# 详细解释
print("\n" + "=" * 60)
print("滑动窗口导致的数据泄露分析")
print("=" * 60)
print("""
原始序列 (用户1): [6846, 7873, 4585, 1, 5406]

train.parquet 存储:
  history = [6846, 7873]
  target = 4585

但 dataset.py 会做滑动窗口 (mode='train'):
  sequence = history + [target] = [6846, 7873, 4585]

  滑动窗口生成:
    sample 1: history=[6846], target=7873
    sample 2: history=[6846, 7873], target=4585

valid.parquet 存储:
  history = [6846, 7873, 4585]
  target = 1

dataset.py (mode='evaluation'):
  sequence = history + [target] = [6846, 7873, 4585, 1]
  生成: history=[6846, 7873, 4585], target=1

问题: 训练时模型见过 history=[6846, 7873, 4585] 且知道下一个是...
      但等等, train的sequence只到4585, 不包括1!

让我重新检查...
""")

# 重新检查: train的滑动窗口能否产生和valid相同的history?
print("\n正确分析:")
same_history_count = 0
for idx, row in valid_df.iterrows():
    user = row['user']
    valid_history = list(row['history'])
    valid_target = row['target']

    train_row = train_df[train_df['user'] == user].iloc[0]
    train_history = list(train_row['history'])
    train_target = train_row['target']

    # train的滑动窗口序列: train_history + [train_target]
    train_full_seq = train_history + [train_target]

    # valid的history是: 原始序列[:-1], 即 train_full_seq + [valid_target][:-1] = train_full_seq
    # 所以 valid_history == train_full_seq!
    if valid_history == train_full_seq:
        same_history_count += 1

print(f"Valid history == Train full sequence 的样本数: {same_history_count}/{len(valid_df)}")
print(f"\n这意味着: 训练时最后一个滑动窗口样本的 history 就是 valid 的 history!")
print(f"但训练时的 target 是 train_target, 不是 valid_target")
print(f"所以理论上没有直接泄露 valid_target")

# 但为什么valid比test好那么多?
print("\n" + "=" * 60)
print("Valid vs Test 难度分析")
print("=" * 60)

# 检查target的流行度
from collections import Counter

all_items = []
for _, row in train_df.iterrows():
    all_items.extend(list(row['history']))
    all_items.append(row['target'])

item_counts = Counter(all_items)

valid_target_counts = [item_counts.get(row['target'], 0) for _, row in valid_df.iterrows()]
test_target_counts = [item_counts.get(row['target'], 0) for _, row in test_df.iterrows()]

print(f"Valid targets 在训练集中的平均出现次数: {np.mean(valid_target_counts):.2f}")
print(f"Test targets 在训练集中的平均出现次数: {np.mean(test_target_counts):.2f}")

# 检查valid和test的target重叠
valid_targets = set(valid_df['target'].values)
test_targets = set(test_df['target'].values)
train_items_set = set(all_items)

print(f"\nValid targets 在训练集中出现过的比例: {len(valid_targets & train_items_set) / len(valid_targets):.2%}")
print(f"Test targets 在训练集中出现过的比例: {len(test_targets & train_items_set) / len(test_targets):.2%}")
