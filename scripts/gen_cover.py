"""
Generate a cover image for GenRec project (Xiaohongshu style).
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Figure setup - 小红书竖版 3:4
fig, ax = plt.subplots(figsize=(9, 12))
ax.set_xlim(0, 9)
ax.set_ylim(0, 12)
ax.axis('off')

# Background gradient (dark blue to purple)
for i in range(1200):
    y = i / 100.0
    r = 0.08 + 0.06 * (y / 12)
    g = 0.08 + 0.02 * (y / 12)
    b = 0.18 + 0.12 * (y / 12)
    ax.axhspan(y, y + 0.01, color=(r, g, b), zorder=0)

# Top area - decorative dots
np.random.seed(42)
for _ in range(60):
    x = np.random.uniform(0, 9)
    y = np.random.uniform(0, 12)
    size = np.random.uniform(0.5, 3)
    alpha = np.random.uniform(0.03, 0.12)
    ax.plot(x, y, 'o', color='#7CB9E8', markersize=size, alpha=alpha, zorder=1)

# Main title
ax.text(4.5, 9.8, 'GenRec', fontsize=72, fontweight='bold',
        color='white', ha='center', va='center', zorder=5,
        fontfamily='DejaVu Sans')

# Subtitle
ax.text(4.5, 8.8, 'A Model Zoo for', fontsize=28,
        color='#B0C4DE', ha='center', va='center', zorder=5,
        fontfamily='DejaVu Sans', fontweight='medium')
ax.text(4.5, 8.1, 'Generative Recommendation', fontsize=28,
        color='#B0C4DE', ha='center', va='center', zorder=5,
        fontfamily='DejaVu Sans', fontweight='medium')

# Divider line
ax.plot([2, 7], [7.3, 7.3], color='#7CB9E8', linewidth=2, alpha=0.5, zorder=5)

# Model cards area
models = [
    ('SASRec', 'Baseline', '#D4A574'),
    ('HSTU', 'Baseline', '#C9C9C9'),
    ('TIGER', 'Generative', '#E8A87C'),
    ('LCRec', 'Generative', '#7CB9E8'),
    ('RQVAE', 'Generative', '#A9A9A9'),
    ('COBRA', 'Generative', '#B39DDB'),
]

# Draw model cards in 2x3 grid
card_w, card_h = 2.2, 1.3
start_x, start_y = 0.9, 4.2
gap_x, gap_y = 2.7, 1.7

for idx, (name, mtype, color) in enumerate(models):
    col = idx % 3
    row = idx // 3
    cx = start_x + col * gap_x
    cy = start_y + (1 - row) * gap_y

    # Card background
    rect = patches.FancyBboxPatch(
        (cx, cy), card_w, card_h,
        boxstyle="round,pad=0.1",
        facecolor=color, alpha=0.2,
        edgecolor=color, linewidth=1.5,
        zorder=3
    )
    ax.add_patch(rect)

    # Model name
    ax.text(cx + card_w/2, cy + card_h*0.6, name,
            fontsize=18, fontweight='bold', color='white',
            ha='center', va='center', zorder=5,
            fontfamily='DejaVu Sans')
    # Type label
    ax.text(cx + card_w/2, cy + card_h*0.25, mtype,
            fontsize=11, color=color, alpha=0.9,
            ha='center', va='center', zorder=5,
            fontfamily='DejaVu Sans')

# Recall@10 mini chart at bottom area
ax.text(4.5, 3.4, 'Recall@10 on 4 Amazon Datasets', fontsize=14,
        color='#B0C4DE', ha='center', va='center', zorder=5,
        fontfamily='DejaVu Sans', fontstyle='italic')

# Mini bar chart
datasets_labels = ['Beauty', 'Sports', 'Toys', 'Arts']
lcrec_vals = [7.9, 4.2, 6.8, 6.3]
tiger_vals = [7.2, 4.1, 6.5, 6.2]
bar_x_start = 1.5
bar_width = 0.25
bar_gap = 1.6
max_val = 9.0
bar_height_scale = 1.8 / max_val

for i, (label, lv, tv) in enumerate(zip(datasets_labels, lcrec_vals, tiger_vals)):
    bx = bar_x_start + i * bar_gap
    by = 1.2

    # TIGER bar
    th = tv * bar_height_scale
    rect_t = patches.FancyBboxPatch(
        (bx, by), bar_width, th,
        boxstyle="round,pad=0.02",
        facecolor='#E8A87C', alpha=0.7, zorder=4
    )
    ax.add_patch(rect_t)

    # LCRec bar
    lh = lv * bar_height_scale
    rect_l = patches.FancyBboxPatch(
        (bx + bar_width + 0.05, by), bar_width, lh,
        boxstyle="round,pad=0.02",
        facecolor='#7CB9E8', alpha=0.7, zorder=4
    )
    ax.add_patch(rect_l)

    # Dataset label
    ax.text(bx + bar_width + 0.025, 0.95, label,
            fontsize=9, color='#888888', ha='center', va='center', zorder=5)

# Bottom - GitHub info
ax.text(4.5, 0.4, 'phonism/genrec  |  SASRec  HSTU  TIGER  LCRec  RQVAE  COBRA',
        fontsize=11, color='#666666', ha='center', va='center', zorder=5,
        fontfamily='DejaVu Sans')

plt.tight_layout(pad=0)

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, '..', 'assets', 'xhs_cover.png')
plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#141428', pad_inches=0.1)
print(f"Saved to {output_path}")
