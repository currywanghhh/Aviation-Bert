import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# -------------------------- 1. 基础配置（完全参考训练损失图代码）--------------------------
# 中文字体设置（与损失图一致，避免中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 模型名称与输出配置（匹配损失图的输出目录和命名规则）
model_name = "self-bert"
output_dir = "output-yuanshi"  # 与损失图输出目录一致
os.makedirs(output_dir, exist_ok=True)  # 确保目录存在

# -------------------------- 2. 加载你提供的指标数据（self-bert_overall_metrics.csv）--------------------------
# 读取CSV文件（请确保文件路径正确，若在output-yuanshi目录下需补充路径）
metrics_df = pd.read_csv('self-bert_overall_metrics.csv')  # 若文件在output-yuanshi内，改为f"{output_dir}/self-bert_overall_metrics.csv"

# 提取关键数据（从CSV中获取epoch和4项指标数值，确保与你的数据完全一致）
epochs = metrics_df['epoch'].tolist()  # 1-20 epoch
accuracy = metrics_df['accuracy'].tolist()  # 准确率数据
precision = metrics_df['precision'].tolist()  # 精确率数据
recall = metrics_df['recall'].tolist()  # 召回率数据
# 修正recall列可能的格式异常（若CSV中存在多余字符，如"0.90321632666121113"，提取纯数值）
recall = [float(str(val).split('|')[0]) if '|' in str(val) else float(val) for val in recall]
f1 = metrics_df['f1'].tolist()  # F1分数数据

# -------------------------- 3. 绘制指标图（参考损失图设计风格）--------------------------
# 画布尺寸：10*6（与损失图plt.figure(figsize=(10, 6))完全一致）
plt.figure(figsize=(10, 6))

# 绘制4项指标趋势线（参考损失图的"marker='o'"和网格风格，区分指标颜色）
plt.plot(epochs, accuracy, marker='o', label='Accuracy')    # 蓝色-准确率
plt.plot(epochs, precision, marker='o', label='Precision')  # 紫色-精确率
plt.plot(epochs, recall, marker='o', label='Recall')        # 橙色-召回率
plt.plot(epochs, f1, marker='o', label='F1')                # 红色-F1分数

# 坐标轴与标题（参考损失图的简洁风格，标签文字完全匹配需求）
plt.xlabel('Epoch')  # 横坐标标签（与损失图"Epoch"一致）
plt.ylabel('Metric Value')  # 纵坐标标签（对应指标值，类比损失图"Loss Value"）
plt.title('Overall Metrics Over Epochs')  # 图表标题（明确指标主题）

# 图例与网格（完全参考损失图配置：显示图例、启用网格）
plt.legend()  # 显示指标图例（类比损失图"Training Loss"图例）
plt.grid(True)  # 启用网格线（与损失图plt.grid(True)一致，辅助读数）

# 布局调整与保存（与损失图的保存逻辑完全一致）
plt.tight_layout()  # 自动调整布局，避免标签截断
# 保存路径：output-yuanshi/模型名_overall_metrics.png（类比损失图的命名格式）
plt.savefig(f'{output_dir}/{model_name}_overall_metrics.png')
plt.close()  # 关闭画布，释放内存

print(f"指标图已保存至：{output_dir}/{model_name}_overall_metrics.png")