import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import json
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from dataset import TextDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from modelscope import snapshot_download
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 自定义 Trainer 类来记录每个 epoch 的指标
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        self.training_loss = []  # 添加训练损失列表

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # 调用原始的 evaluate 方法获取指标结果
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # 记录本次 epoch 的指标
        self.epoch_metrics['accuracy'].append(metrics.get(f"{metric_key_prefix}_accuracy", 0))
        self.epoch_metrics['precision'].append(metrics.get(f"{metric_key_prefix}_precision", 0))
        self.epoch_metrics['recall'].append(metrics.get(f"{metric_key_prefix}_recall", 0))
        self.epoch_metrics['f1'].append(metrics.get(f"{metric_key_prefix}_f1", 0))
        
        # 记录最近的训练损失
        if self.state.log_history:
            recent_loss = next((log['loss'] for log in reversed(self.state.log_history) 
                              if 'loss' in log), None)
            if recent_loss is not None:
                self.training_loss.append(recent_loss)
        
        return metrics

def load_data(file_path):
    df = pd.read_excel(file_path)
    texts = df['text'].tolist()
    labels = [label.replace('\n', '') for label in df['label'].tolist()]
    return texts, labels

def preprocess_data(texts, labels):
    # 标签编码
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    # 保存标签编码以便后续使用
    label_mapping = {index: label for index, label in enumerate(le.classes_)}
    with open('data/label_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=4)

    # 数据拆分
    X_train, X_val, y_train, y_val = train_test_split(texts, labels_encoded, test_size=0.2, random_state=42)
    
    return X_train, X_val, y_train, y_val

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def plot_overall_metrics(trainer, model_name):
    """绘制训练过程中的整体指标图表"""
    os.makedirs('output-yuanshi', exist_ok=True)
    
    # 绘制评估指标图表
    plt.figure(figsize=(10, 6))
    epochs = list(range(1, len(trainer.epoch_metrics['accuracy']) + 1))
    
    plt.plot(epochs, trainer.epoch_metrics['accuracy'], marker='o', label='Accuracy')
    plt.plot(epochs, trainer.epoch_metrics['precision'], marker='o', label='Precision')
    plt.plot(epochs, trainer.epoch_metrics['recall'], marker='o', label='Recall')
    plt.plot(epochs, trainer.epoch_metrics['f1'], marker='o', label='F1 Score')
    
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Evaluation Metrics Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'output-yuanshi/{model_name}_overall_metrics.png')
    plt.close()
    
    # 绘制训练损失图表
    plt.figure(figsize=(10, 6))
    loss_epochs = list(range(1, len(trainer.training_loss) + 1))
    plt.plot(loss_epochs, trainer.training_loss, marker='o', color='red', label='Training Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'output-yuanshi/{model_name}_training_loss.png')
    plt.close()
    
    # 保存所有指标数据到CSV
    metrics_df = pd.DataFrame({
        'epoch': epochs,
        'accuracy': trainer.epoch_metrics['accuracy'],
        'precision': trainer.epoch_metrics['precision'],
        'recall': trainer.epoch_metrics['recall'],
        'f1': trainer.epoch_metrics['f1'],
        'training_loss': trainer.training_loss if len(trainer.training_loss) >= len(epochs) else trainer.training_loss + [None] * (len(epochs) - len(trainer.training_loss))
    })
    metrics_df.to_csv(f'output-yuanshi/{model_name}_metrics_by_epoch.csv', index=False)
    
    # 单独保存训练损失数据到CSV
    loss_df = pd.DataFrame({
        'epoch': loss_epochs,
        'training_loss': trainer.training_loss
    })
    loss_df.to_csv(f'output-yuanshi/{model_name}_training_loss.csv', index=False)

def train(model_name, model_path):
    texts, labels = load_data('data/备注和事件分类全.xlsx')
    X_train, X_val, y_train, y_val = preprocess_data(texts, labels)

    if model_name == 'self-bert':
        # 使用原有的模型和tokenizer
        tokenizer = BertTokenizer.from_pretrained('pretrained-models/update-toke-bert-base-chinese')
        model = BertForSequenceClassification.from_pretrained('pretrained-models/self-bert-pretrain', num_labels=len(set(labels)))
    else:
        # 使用 ModelScope 下载模型到指定目录
        local_model_dir = snapshot_download(model_path, cache_dir='pretrained-models')
        tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(local_model_dir, num_labels=len(set(labels)))

    train_dataset = TextDataset(X_train, y_train, tokenizer, max_len=128)
    val_dataset = TextDataset(X_val, y_val, tokenizer, max_len=128)

    training_args = TrainingArguments(
        output_dir=f'models/{model_name}-classification',
        overwrite_output_dir=True,
        num_train_epochs=20,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",  # 每个 epoch 评估一次
        save_total_limit=3,
        logging_dir=f'models/{model_name}-classification-logs',
    )

    # 使用自定义的 Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    
    # 绘制整体指标图表
    plot_overall_metrics(trainer, model_name)
    
    trainer.save_model(f'models/{model_name}-classification')

if __name__ == "__main__":
    # 定义要训练的模型
    models = {
        'self-bert': None,  # 原有的模型
        # 'chinese-roberta-wwm-ext': 'dienstag/chinese-roberta-wwm-ext',
        # 'chinese-macbert-base': 'dienstag/chinese-macbert-base'
    }

    # 依次训练每个模型
    for model_name, model_path in models.items():
        print(f"开始训练模型: {model_name}")
        train(model_name, model_path)
        print(f"完成模型 {model_name} 的训练")
