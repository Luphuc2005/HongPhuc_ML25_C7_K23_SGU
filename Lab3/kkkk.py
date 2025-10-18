import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# Tạo figure
fig, ax = plt.subplots(1, 1, figsize=(20, 16))
ax.set_xlim(0, 20)
ax.set_ylim(0, 16)
ax.axis('off')

# Màu sắc
colors = {
    'main': '#FF6B6B',
    'eda': '#4ECDC4', 
    'engineering': '#45B7D1',
    'preprocessing': '#96CEB4',
    'modeling': '#FFEAA7',
    'evaluation': '#DDA0DD'
}

# 1. Trung tâm - Mục tiêu chính
center_box = FancyBboxPatch((9, 7.5), 2, 1.5, 
                           boxstyle="round,pad=0.1", 
                           facecolor=colors['main'], 
                           edgecolor='black', linewidth=3)
ax.add_patch(center_box)
ax.text(10, 8.25, 'Titanic\nSurvival\nPrediction', ha='center', va='center', 
        fontsize=14, fontweight='bold', color='white')

# 2. EDA - Khảo sát dữ liệu
eda_box = FancyBboxPatch((2, 11), 3, 2, 
                        boxstyle="round,pad=0.1", 
                        facecolor=colors['eda'], 
                        edgecolor='black', linewidth=2)
ax.add_patch(eda_box)
ax.text(3.5, 12, 'EDA\n(Exploratory\nData Analysis)', ha='center', va='center', 
        fontsize=12, fontweight='bold')

# Chi tiết EDA
ax.text(3.5, 11.3, '• Missing values\n• Correlations\n• Distributions\n• Class imbalance\n• Outliers', 
        ha='center', va='top', fontsize=9)

# 3. Feature Engineering
fe_box = FancyBboxPatch((15, 11), 3, 2, 
                        boxstyle="round,pad=0.1", 
                        facecolor=colors['engineering'], 
                        edgecolor='black', linewidth=2)
ax.add_patch(fe_box)
ax.text(16.5, 12, 'Feature\nEngineering', ha='center', va='center', 
        fontsize=12, fontweight='bold')

# Chi tiết Feature Engineering
ax.text(16.5, 11.3, '• Family size\n• Title extraction\n• Deck from Cabin\n• Ticket frequency\n• Interaction features', 
        ha='center', va='top', fontsize=9)

# 4. Preprocessing
prep_box = FancyBboxPatch((2, 6), 3, 2, 
                          boxstyle="round,pad=0.1", 
                          facecolor=colors['preprocessing'], 
                          edgecolor='black', linewidth=2)
ax.add_patch(prep_box)
ax.text(3.5, 7, 'Data\nPreprocessing', ha='center', va='center', 
        fontsize=12, fontweight='bold')

# Chi tiết Preprocessing
ax.text(3.5, 6.3, '• Handle missing\n• Encode categorical\n• Scale features\n• Feature selection', 
        ha='center', va='top', fontsize=9)

# 5. Modeling
model_box = FancyBboxPatch((15, 6), 3, 2, 
                           boxstyle="round,pad=0.1", 
                           facecolor=colors['modeling'], 
                           edgecolor='black', linewidth=2)
ax.add_patch(model_box)
ax.text(16.5, 7, 'Modeling', ha='center', va='center', 
        fontsize=12, fontweight='bold')

# Chi tiết Modeling
ax.text(16.5, 6.3, '• Random Forest\n• XGBoost\n• SVM\n• Ensemble', 
        ha='center', va='top', fontsize=9)

# 6. Evaluation
eval_box = FancyBboxPatch((8.5, 2), 3, 2, 
                          boxstyle="round,pad=0.1", 
                          facecolor=colors['evaluation'], 
                          edgecolor='black', linewidth=2)
ax.add_patch(eval_box)
ax.text(10, 3, 'Evaluation\n& Validation', ha='center', va='center', 
        fontsize=12, fontweight='bold')

# Chi tiết Evaluation
ax.text(10, 2.3, '• K-Fold CV\n• GridSearch\n• Feature importance\n• ROC/AUC', 
        ha='center', va='top', fontsize=9)

# 7. Mũi tên kết nối từ trung tâm
# Trung tâm → EDA
arrow1 = ConnectionPatch((9, 9), (3.5, 11), "data", "data",
                        arrowstyle="->", shrinkA=5, shrinkB=5, 
                        mutation_scale=20, fc="black", linewidth=2)
ax.add_patch(arrow1)

# Trung tâm → Feature Engineering
arrow2 = ConnectionPatch((11, 9), (16.5, 11), "data", "data",
                        arrowstyle="->", shrinkA=5, shrinkB=5, 
                        mutation_scale=20, fc="black", linewidth=2)
ax.add_patch(arrow2)

# Trung tâm → Preprocessing
arrow3 = ConnectionPatch((9, 7.5), (3.5, 8), "data", "data",
                        arrowstyle="->", shrinkA=5, shrinkB=5, 
                        mutation_scale=20, fc="black", linewidth=2)
ax.add_patch(arrow3)

# Trung tâm → Modeling
arrow4 = ConnectionPatch((11, 7.5), (16.5, 8), "data", "data",
                        arrowstyle="->", shrinkA=5, shrinkB=5, 
                        mutation_scale=20, fc="black", linewidth=2)
ax.add_patch(arrow4)

# Trung tâm → Evaluation
arrow5 = ConnectionPatch((10, 7.5), (10, 4), "data", "data",
                        arrowstyle="->", shrinkA=5, shrinkB=5, 
                        mutation_scale=20, fc="black", linewidth=2)
ax.add_patch(arrow5)

# 8. Mũi tên kết nối giữa các bước
# EDA → Preprocessing
arrow6 = ConnectionPatch((3.5, 11), (3.5, 8), "data", "data",
                        arrowstyle="->", shrinkA=5, shrinkB=5, 
                        mutation_scale=15, fc="gray", alpha=0.7)
ax.add_patch(arrow6)

# Feature Engineering → Modeling
arrow7 = ConnectionPatch((16.5, 11), (16.5, 8), "data", "data",
                        arrowstyle="->", shrinkA=5, shrinkB=5, 
                        mutation_scale=15, fc="gray", alpha=0.7)
ax.add_patch(arrow7)

# Preprocessing → Evaluation
arrow8 = ConnectionPatch((5.5, 7), (8.5, 3), "data", "data",
                        arrowstyle="->", shrinkA=5, shrinkB=5, 
                        mutation_scale=15, fc="gray", alpha=0.7)
ax.add_patch(arrow8)

# Modeling → Evaluation
arrow9 = ConnectionPatch((15, 7), (11.5, 3), "data", "data",
                        arrowstyle="->", shrinkA=5, shrinkB=5, 
                        mutation_scale=15, fc="gray", alpha=0.7)
ax.add_patch(arrow9)

# 9. Thêm các ý tưởng phụ
# Data Quality
quality_box = FancyBboxPatch((0.5, 13.5), 2, 1, 
                             boxstyle="round,pad=0.05", 
                             facecolor='#FFE4E1', 
                             edgecolor='black', linewidth=1)
ax.add_patch(quality_box)
ax.text(1.5, 14, 'Data Quality\nIssues', ha='center', va='center', fontsize=10)

# Domain Knowledge
domain_box = FancyBboxPatch((17.5, 13.5), 2, 1, 
                            boxstyle="round,pad=0.05", 
                            facecolor='#E1F5FE', 
                            edgecolor='black', linewidth=1)
ax.add_patch(domain_box)
ax.text(18.5, 14, 'Domain\nKnowledge', ha='center', va='center', fontsize=10)

# 10. Legend
legend_elements = [
    mpatches.Patch(color=colors['main'], label='Main Goal'),
    mpatches.Patch(color=colors['eda'], label='EDA'),
    mpatches.Patch(color=colors['engineering'], label='Feature Engineering'),
    mpatches.Patch(color=colors['preprocessing'], label='Preprocessing'),
    mpatches.Patch(color=colors['modeling'], label='Modeling'),
    mpatches.Patch(color=colors['evaluation'], label='Evaluation')
]

ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), 
          fontsize=10, framealpha=0.8)

# 11. Title
ax.text(10, 15, 'Titanic Survival Prediction - Mind Map', 
        ha='center', va='center', fontsize=18, fontweight='bold')

# 12. Thêm text mô tả
ax.text(10, 0.5, 'Key Insight: Combine domain knowledge with data-driven feature engineering\nfor robust survival prediction', 
        ha='center', va='center', fontsize=11, style='italic', 
        bbox=dict(boxstyle="round,pad=0.3", facecolor='#F0F8FF', alpha=0.8))

plt.tight_layout()
plt.show()