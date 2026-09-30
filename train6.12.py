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

# 定义小样本类别
SMALL_SAMPLE_CLASSES = [
    "落地超重/速度大等原因",
    "紧急事件通报：收到飞行机组必需成员能力丧失信息。",
    "旅客服务",
    "紧急事件通报：跑道侵入",
    "紧急事件通报：管制反馈无法与机组建立联系",
    "雷击",
    "紧急事件通报：航空器在地面出现非动力位移",
    "紧急事件通报：空中出现明火、烟雾或火警/烟雾信息，机组判断对飞行有影响。",
    "无人机、不明飞行物",
    "紧急事件通报：在运行阶段航空器受损或人员受伤、死亡。",
    "紧急事件通报：飞错航路、飞偏或飞错进离场航线、未正确执行复飞程序并导致其他航空器避让 （例如：调整高度、调整航向、调整航路）。",
    "紧急事件通报：航空器爆炸物威胁",
    "紧急事件通报：起飞和着陆过程中，冲出、偏出跑道或跑道外接地。注：含中断起飞停在停止道的情形。",
    "紧急事件通报：航空器失去一切联系0至30 分钟，且无法确认飞机位置",
    "系统问题",
    "紧急事件通报：机上有可疑传染病病例",
    "紧急事件通报：地面航空器在起火",
    "紧急事件通报：航空器上发现爆炸物或其他危险物品，可能危害飞行安全",
    "紧急事件通报：低于运行标准 （机场运行最低标准、航空器运行标准、飞行机组资格标准）起飞、开始最后进近或着陆",
    "紧急事件通报：起落架机轮 （滑橇、尾环、浮筒）之外的任何部位触地/水",
    "紧急事件通报：出现烟雾或毒气等需要飞行机组成员使用氧气的紧急情况",
    "紧急事件通报：飞行员宣布紧急状态'MAYDAY'",
    "紧急事件通报：机场受到爆炸物威胁",
    "燃油低温"
]

# 全局标签编码器
le = LabelEncoder()


# 自定义 Trainer 类来记录每个 epoch 的指标
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'small_sample_accuracy': [],
            'small_sample_precision': [],
            'small_sample_recall': [],
            'small_sample_f1': []
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
        self.epoch_metrics['small_sample_accuracy'].append(metrics.get(f"{metric_key_prefix}_small_sample_accuracy", 0))
        self.epoch_metrics['small_sample_precision'].append(metrics.get(f"{metric_key_prefix}_small_sample_precision", 0))
        self.epoch_metrics['small_sample_recall'].append(metrics.get(f"{metric_key_prefix}_small_sample_recall", 0))
        self.epoch_metrics['small_sample_f1'].append(metrics.get(f"{metric_key_prefix}_small_sample_f1", 0))
        
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
    # 使用全局的标签编码器
    global le
    labels_encoded = le.fit_transform(labels)

    # 保存标签编码映射
    label_mapping = {index: label for index, label in enumerate(le.classes_)}
    with open('data/label_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=4)

    # 分离小样本和大样本数据
    small_sample_indices = [i for i, label in enumerate(labels) if label in SMALL_SAMPLE_CLASSES]
    large_sample_indices = [i for i, label in enumerate(labels) if label not in SMALL_SAMPLE_CLASSES]
    
    # 分别获取小样本和大样本数据
    small_texts = [texts[i] for i in small_sample_indices]
    small_labels = [labels_encoded[i] for i in small_sample_indices]
    large_texts = [texts[i] for i in large_sample_indices]
    large_labels = [labels_encoded[i] for i in large_sample_indices]

    # 分别进行训练集和测试集划分
    small_X_train, small_X_val, small_y_train, small_y_val = train_test_split(
        small_texts, small_labels, test_size=0.2, random_state=42
    )
    large_X_train, large_X_val, large_y_train, large_y_val = train_test_split(
        large_texts, large_labels, test_size=0.2, random_state=42
    )

    # 合并训练集和测试集
    X_train = small_X_train + large_X_train
    X_val = small_X_val + large_X_val
    y_train = small_y_train + large_y_train
    y_val = small_y_val + large_y_val

    return X_train, X_val, y_train, y_val, small_sample_indices

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # 计算整体指标
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    # 计算小样本数据的指标
    small_sample_mask = np.isin(predictions, [i for i, label in enumerate(le.classes_) if label in SMALL_SAMPLE_CLASSES])
    if np.any(small_sample_mask):
        small_sample_accuracy = accuracy_score(labels[small_sample_mask], predictions[small_sample_mask])
        small_sample_precision, small_sample_recall, small_sample_f1, _ = precision_recall_fscore_support(
            labels[small_sample_mask], 
            predictions[small_sample_mask], 
            average='weighted'
        )
    else:
        small_sample_accuracy = 0
        small_sample_precision = 0
        small_sample_recall = 0
        small_sample_f1 = 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'small_sample_accuracy': small_sample_accuracy,
        'small_sample_precision': small_sample_precision,
        'small_sample_recall': small_sample_recall,
        'small_sample_f1': small_sample_f1
    }

def plot_overall_metrics(trainer, model_name):
    """绘制训练过程中的整体指标图表"""
    os.makedirs('output-yuanshi', exist_ok=True)
    epochs = list(range(1, len(trainer.epoch_metrics['accuracy']) + 1))
    
    # 1. 绘制总体评估指标图表
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, trainer.epoch_metrics['accuracy'], marker='o', label='Accuracy')
    plt.plot(epochs, trainer.epoch_metrics['precision'], marker='o', label='Precision')
    plt.plot(epochs, trainer.epoch_metrics['recall'], marker='o', label='Recall')
    plt.plot(epochs, trainer.epoch_metrics['f1'], marker='o', label='F1 Score')
    
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Overall Metrics Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'output-yuanshi/{model_name}_overall_metrics.png')
    plt.close()
    
    # 保存总体评估指标数据
    overall_metrics_df = pd.DataFrame({
        'epoch': epochs,
        'accuracy': trainer.epoch_metrics['accuracy'],
        'precision': trainer.epoch_metrics['precision'],
        'recall': trainer.epoch_metrics['recall'],
        'f1': trainer.epoch_metrics['f1']
    })
    overall_metrics_df.to_csv(f'output-yuanshi/{model_name}_overall_metrics.csv', index=False)
    
    # 2. 绘制小样本评估指标图表
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, trainer.epoch_metrics['small_sample_accuracy'], marker='o', label='Accuracy')
    plt.plot(epochs, trainer.epoch_metrics['small_sample_precision'], marker='o', label='Precision')
    plt.plot(epochs, trainer.epoch_metrics['small_sample_recall'], marker='o', label='Recall')
    plt.plot(epochs, trainer.epoch_metrics['small_sample_f1'], marker='o', label='F1 Score')
    
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Small Sample Metrics Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'output-yuanshi/{model_name}_small_sample_metrics.png')
    plt.close()
    
    # 保存小样本评估指标数据
    small_sample_metrics_df = pd.DataFrame({
        'epoch': epochs,
        'accuracy': trainer.epoch_metrics['small_sample_accuracy'],
        'precision': trainer.epoch_metrics['small_sample_precision'],
        'recall': trainer.epoch_metrics['small_sample_recall'],
        'f1': trainer.epoch_metrics['small_sample_f1']
    })
    small_sample_metrics_df.to_csv(f'output-yuanshi/{model_name}_small_sample_metrics.csv', index=False)
    
    # 3. 绘制训练损失图表
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
    
    # 保存训练损失数据
    loss_df = pd.DataFrame({
        'epoch': loss_epochs,
        'training_loss': trainer.training_loss
    })
    loss_df.to_csv(f'output-yuanshi/{model_name}_training_loss.csv', index=False)

def train(model_name, model_path):
    texts, labels = load_data('data/增强备注和事件分类全.xlsx')
    X_train, X_val, y_train, y_val, small_sample_indices = preprocess_data(texts, labels)

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
