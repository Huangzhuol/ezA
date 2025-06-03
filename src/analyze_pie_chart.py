import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.patches import Rectangle


plt.style.use('seaborn-v0_8')
plt.rcParams['font.size'] = 12


rag_scores = {1: 5, 2: 10, 3: 15, 4: 40, 5: 30}
non_rag_scores = {1: 2, 2: 8, 3: 20, 4: 45, 5: 25}


labels = ['Score 1', 'Score 2', 'Score 3', 'Score 4', 'Score 5']
colors = ['#ff6b6b', '#ffa3a3', '#66b3ff', '#99ff99', '#2ecc71']
rag_counts = [rag_scores[i] for i in range(1,6)]
non_rag_counts = [non_rag_scores[i] for i in range(1,6)]


fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.5], height_ratios=[0.1, 1])


title_ax = fig.add_subplot(gs[0, :])
title_ax.set_title('Score Distribution Comparison: RAG vs Non-RAG Models', fontsize=16, pad=20)
title_ax.axis('off')


ax1 = fig.add_subplot(gs[1, 0])
wedges1, _ = ax1.pie(
    rag_counts,
    colors=colors,
    startangle=90,
    wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
    radius=0.6,
    pctdistance=0.8
)


for i, wedge in enumerate(wedges1):
    if rag_counts[i] == 0:
        continue
        
    angle = (wedge.theta2 - wedge.theta1)/2. + wedge.theta1
    x = 0.8 * np.cos(np.deg2rad(angle))
    y = 0.8 * np.sin(np.deg2rad(angle))
    
    
    offset_x = 1.3 if x > 0 else -1.3
    offset_y = 1.3 if y > 0 else -1.3
    
    ax1.annotate(
        f"{labels[i]}\n{rag_counts[i]} ({rag_counts[i]/sum(rag_counts):.1%})",
        xy=(x, y),
        xytext=(offset_x, offset_y),
        textcoords='offset points',
        ha='center',
        va='center',
        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5)
    )

ax1.set_title('RAG Models', pad=20)
ax2 = fig.add_subplot(gs[1, 1])
wedges2, _ = ax2.pie(
    non_rag_counts,
    colors=colors,
    startangle=90,
    wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
    radius=0.6,
    pctdistance=0.8
)


for i, wedge in enumerate(wedges2):
    if non_rag_counts[i] == 0:
        continue
        
    angle = (wedge.theta2 - wedge.theta1)/2. + wedge.theta1
    x = 0.8 * np.cos(np.deg2rad(angle))
    y = 0.8 * np.sin(np.deg2rad(angle))
    
    
    offset_x = 1.3 if x > 0 else -1.3
    offset_y = 1.3 if y > 0 else -1.3
    
    ax2.annotate(
        f"{labels[i]}\n{non_rag_counts[i]} ({non_rag_counts[i]/sum(non_rag_counts):.1%})",
        xy=(x, y),
        xytext=(offset_x, offset_y),
        textcoords='offset points',
        ha='center',
        va='center',
        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5)
    )

ax2.set_title('Non-RAG Models', pad=20)


table_ax = fig.add_subplot(gs[1, 2])
table_ax.axis('off')


table_data = []
for i in range(5):
    rag_pct = rag_counts[i]/sum(rag_counts)*100 if sum(rag_counts) > 0 else 0
    non_rag_pct = non_rag_counts[i]/sum(non_rag_counts)*100 if sum(non_rag_counts) > 0 else 0
    table_data.append([
        labels[i],
        f"{rag_counts[i]} ({rag_pct:.1f}%)",
        f"{non_rag_counts[i]} ({non_rag_pct:.1f}%)"
    ])


table = table_ax.table(
    cellText=table_data,
    colLabels=['Score', 'RAG', 'Non-RAG'],
    loc='center',
    cellLoc='center',
    colColours=['#f0f0f0', '#e6f7ff', '#fff2e6']
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.5)

plt.tight_layout()
os.makedirs('./result', exist_ok=True)
plt.savefig('./result/final_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("saved as: ./result/final_comparison.png")