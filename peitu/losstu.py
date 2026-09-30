import matplotlib.pyplot as plt

# 数据
epochs = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19
]
training_loss = [
    1.9004, 1.9004, 0.5495, 0.2905, 0.2905, 0.1605, 0.1605, 0.1278, 0.1042, 0.1042,
    0.0833, 0.0833, 0.1047, 0.0757, 0.0757, 0.0669, 0.0514, 0.0514, 0.0489
]

# 画布尺寸与字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10, 6))

# 绘制红色训练损失曲线
plt.plot(epochs, training_loss, marker='o', color='#d62728', label='Training Loss')

# 坐标轴与标题
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.title('Training Loss Over Epochs')

# 图例与网格
plt.legend()
plt.grid(True)

# 布局调整与保存
plt.tight_layout()
plt.savefig('self-bert_training_loss.png')
plt.close()

print("训练损失图已保存至：self-bert_training_loss.png")