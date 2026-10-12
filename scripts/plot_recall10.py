"""
Plot Recall@10 comparison across models and datasets.
Dark theme version for cover.png embedding.
"""
import matplotlib.pyplot as plt
import numpy as np

# Dark theme
BG_COLOR = '#1a1a2e'
TEXT_COLOR = '#e0e0e0'
GRID_COLOR = '#333355'

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = BG_COLOR
plt.rcParams['axes.facecolor'] = BG_COLOR
plt.rcParams['text.color'] = TEXT_COLOR
plt.rcParams['axes.labelcolor'] = TEXT_COLOR
plt.rcParams['xtick.color'] = TEXT_COLOR
plt.rcParams['ytick.color'] = TEXT_COLOR

# Data from README
datasets = ['Beauty', 'Sports', 'Toys', 'Home']
models = ['SASRec(CE)', 'SASRec(BCE)', 'HSTU']

# Recall@10 values (convert to percentage) - TEST set
recall10 = {
    'SASRec(CE)':  [8.51, 4.95, 9.22, 0],
    'SASRec(BCE)': [5.03, 2.91, 5.94, 0],
    'HSTU':        [8.59, 4.39, 9.14, 0],
}

# Colors matching legend: pink, orange, cyan
colors = ['#FF6B8A', '#FFA64D', '#4ECDC4']
edgecolors = ['#FF4D6D', '#FF8C1A', '#38B2AC']

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(datasets))
width = 0.16
multiplier = 0

for i, (model, values) in enumerate(recall10.items()):
    offset = width * multiplier
    # Replace 0 with NaN so bars don't show for missing data
    plot_values = [v if v > 0 else np.nan for v in values]
    bars = ax.bar(x + offset, plot_values, width,
                  label=model,
                  color=colors[i],
                  edgecolor=edgecolors[i],
                  linewidth=0.8)
    # Add value labels on bars
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9,
                    fontweight='medium', color=TEXT_COLOR)
    multiplier += 1

# Customize
ax.set_ylabel('Recall@10 (%)', fontsize=13, fontweight='medium')
ax.set_xlabel('')
ax.set_xticks(x + width * 1.0)
ax.set_xticklabels(datasets, fontsize=12, fontweight='medium')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=10,
          frameon=True, fancybox=True, facecolor='#252545', edgecolor='#444466',
          labelcolor=TEXT_COLOR)
ax.set_ylim(1, 10)
ax.set_yticks(np.arange(1, 11, 1))

# Style
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color(GRID_COLOR)
ax.spines['bottom'].set_color(GRID_COLOR)
ax.yaxis.grid(True, linestyle='--', alpha=0.3, color=GRID_COLOR)
ax.xaxis.grid(False)

plt.tight_layout()
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, '..', 'assets', 'recall10_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=BG_COLOR)
print(f"Saved to {output_path}")
