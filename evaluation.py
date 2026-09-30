import os
import matplotlib.pyplot as plt
import json

# 定义模型名称和对应的日志文件路径
models = {
    'self-bert': 'models/self-bert-classification/checkpoint-4500/trainer_state.json',
    'chinese-roberta-wwm-ext': 'models/chinese-roberta-wwm-ext-classification/checkpoint-4500/trainer_state.json',
    'chinese-macbert-base': 'models/chinese-macbert-base-classification/checkpoint-4500/trainer_state.json'
}

for model_name, log_file in models.items():
    # 读取日志文件
    with open(log_file, 'r') as f:
        log_data = json.load(f)

    # 提取数据
    epochs = []
    accuracy = []
    precision = []
    recall = []
    f1 = []
    train_losses = []
    eval_losses = []
    train_epochs = []

    # 假设每个epoch的步数是固定的，需要根据实际情况调整
    total_steps = len([entry for entry in log_data['log_history'] if 'loss' in entry])
    last_epoch = log_data['log_history'][-1]['epoch'] if log_data['log_history'] else 0
    steps_per_epoch = total_steps / last_epoch if last_epoch > 0 else 1

    step_counter = 0
    for entry in log_data['log_history']:
        if 'eval_accuracy' in entry:
            epochs.append(entry['epoch'])
            accuracy.append(entry['eval_accuracy'])
            precision.append(entry['eval_precision'])
            recall.append(entry['eval_recall'])
            f1.append(entry['eval_f1'])
        if 'loss' in entry:
            train_losses.append(entry['loss'])
            step_counter += 1
            train_epoch = step_counter / steps_per_epoch
            train_epochs.append(train_epoch)
        if 'eval_loss' in entry:
            eval_losses.append(entry['eval_loss'])

    # 创建输出目录
    output_dir = f'output/{model_name}'
    os.makedirs(output_dir, exist_ok=True)

    # 打印最终的训练和评估损失值
    print(f"模型 {model_name} 的最终训练损失值：", train_losses[-1] if train_losses else "无")
    print(f"模型 {model_name} 的最终评估损失值：", eval_losses[-1] if eval_losses else "无")
    print(f"模型 {model_name} 的最终评估指标：")
    print("Accuracy:", accuracy[-1] if accuracy else "无")
    print("Precision:", precision[-1] if precision else "无")
    print("Recall:", recall[-1] if recall else "无")
    print("F1 Score:", f1[-1] if f1 else "无")

    # 绘制评估指标趋势图
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, accuracy, marker='o', label='Accuracy')
    plt.plot(epochs, precision, marker='o', label='Precision')
    plt.plot(epochs, recall, marker='o', label='Recall')
    plt.plot(epochs, f1, marker='o', label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title(f'{model_name} Evaluation Metrics Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/metrics_trend.png')
    plt.close()

    # 绘制训练损失值趋势图
    plt.figure(figsize=(12, 8))
    plt.plot(train_epochs, train_losses, marker='o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title(f'{model_name} Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/training_loss_trend.png')
    plt.close()

    # 绘制评估损失值趋势图
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, eval_losses, marker='o', label='Evaluation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title(f'{model_name} Evaluation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/evaluation_loss_trend.png')
    plt.close()