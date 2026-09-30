import numpy as np
import matplotlib.pyplot as plt

# -------------------------- 1. 数据准备 --------------------------
# 模型名称（增加到6个）
models = [
    "BERT-base-Chinese",
    "BERT + Vocabulary Adaptation",
    "AviaBERT-base",
    "AviaBERT-base\n+ Data Augmentation",
    "AviaBERT-base\n+ Weighted Loss",
    "AviaBERT"
]

# 6个模型的4项指标数据（顺序：accuracy, precision, recall, F1）
metrics_data = np.array([
    # BERT-base-Chinese
    [0.051, 0.033, 0.0521, 0.0404],
    # BERT + Vocabulary Adaptation
    [0.2840, 0.1850, 0.2910, 0.2215],
    # baseline
    [0.501, 0.333, 0.5115, 0.4036],
    # Data Augmentation
    [0.8636, 0.8059, 0.8626, 0.8333],
    # Weighted Loss
    [0.5218, 0.3905, 0.5175, 0.4451],
    # AviaBERT
    [0.9028, 0.8162, 0.8966, 0.8543]
])

# 横坐标指标名称
metrics = ["Accuracy", "Precision", "Recall", "F1"]

# -------------------------- 2. 图表配置 --------------------------
# 设置字体为Times New Roman
plt.rcParams["font.family"] = ["Times New Roman", "SimHei"]  # 英文用Times New Roman，中文用SimHei
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 使用学术风颜色（增加到6个颜色）
colors = ['#D4A589', '#C47F9A', '#749F82', '#61A0A8', '#9172EC', '#E57373']
bar_width = 0.13  # 调小柱宽，因为现在是6个模型
x = np.arange(len(metrics))
offsets = [-2.5*bar_width, -1.5*bar_width, -0.5*bar_width,
           0.5*bar_width, 1.5*bar_width, 2.5*bar_width]  # 6个位置

# 创建画布
fig, ax = plt.subplots(figsize=(12, 6))  # 稍微加宽画布

# -------------------------- 3. 绘制柱状图 --------------------------
for i, (model, data, color, offset) in enumerate(zip(models, metrics_data, colors, offsets)):
    bar_positions = x + offset
    bars = ax.bar(
        bar_positions, 
        data, 
        width=bar_width, 
        label=model, 
        color=color, 
        edgecolor='white',
        linewidth=1,
        alpha=0.8
    )
    
    # 柱子顶部添加数值标签
    for bar, value in zip(bars, data):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.01,
            f'{value:.2%}',
            ha='center', va='bottom',
            fontsize=7,  # 稍微调小字体
            color='black'
        )

# -------------------------- 4. 图表美化 --------------------------
ax.set_xlabel('Evaluation Metrics', fontsize=12, fontweight='bold', labelpad=6)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=10)

ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold', labelpad=10)
ax.set_ylim(0, 1.0)
yticks_vals = np.arange(0, 1.01, 0.1)
ax.set_yticks(yticks_vals)
ax.set_yticklabels([f'{y:.0%}' for y in yticks_vals])
ax.grid(True, axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

ax.set_title(
    'Comparison of Six Models on Different Evaluation Metrics',
    fontsize=14,
    fontweight='bold',
    pad=20
)

ax.legend(
    loc='upper center', 
    bbox_to_anchor=(0.5, -0.12),  # 稍微下移，因为图例变长了
    ncol=6,
    fontsize=10,
    frameon=True,
    shadow=True
)

plt.tight_layout()

# -------------------------- 5. 保存图表 --------------------------
plt.savefig('model_metrics_comparison.png', dpi=600, bbox_inches='tight')
plt.savefig('model_metrics_comparison.pdf', bbox_inches='tight')

plt.show()