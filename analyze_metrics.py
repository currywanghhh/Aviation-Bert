import json
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertTokenizer, AutoModelForSequenceClassification, Trainer
from dataset import TextDataset
import numpy as np

def load_label_mapping(file_path='data/label_mapping.json'):
    """加载标签映射"""
    with open(file_path, 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)
    return {int(k): v for k, v in label_mapping.items()}

def load_data_and_model(model_name, checkpoint_path):
    """加载数据和模型"""
    # 加载原始数据
    df = pd.read_excel('data/备注和事件分类全.xlsx')
    texts = df['text'].tolist()
    labels = [label.replace('\n', '') for label in df['label'].tolist()]
    
    # 加载标签映射
    label_mapping = load_label_mapping()
    
    # 修正路径问题
    model_path = f'models/{model_name}-classification/{checkpoint_path}'
    print(f"尝试加载模型，路径: {model_path}")
    
    # 检查路径是否存在
    if not os.path.exists(model_path):
        print(f"错误: 路径 {model_path} 不存在!")
        # 尝试列出可能的路径
        parent_dir = f'models/{model_name}-classification'
        if os.path.exists(parent_dir):
            print(f"可用的检查点:")
            for item in os.listdir(parent_dir):
                if os.path.isdir(os.path.join(parent_dir, item)):
                    print(f"  - {item}")
        
        # 尝试不同的路径格式
        alt_model_path = f'models/{model_name}-classification'
        if os.path.exists(alt_model_path):
            print(f"尝试使用替代路径: {alt_model_path}")
            model_path = alt_model_path
    
    # 加载分词器和模型
    try:
        # 使用指定的分词器路径
        tokenizer_path = 'pretrained-models/dienstag/chinese-macbert-base'
        print(f"尝试加载分词器，路径: {tokenizer_path}")
        
        if os.path.exists(tokenizer_path):
            tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
            print("成功加载分词器")
        else:
            print(f"错误: 分词器路径 {tokenizer_path} 不存在!")
            raise FileNotFoundError(f"分词器路径 {tokenizer_path} 不存在")
        
        # 加载模型
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        print("成功加载模型")
    except Exception as e:
        print(f"加载失败，错误: {e}")
        raise
    
    return texts, labels, label_mapping, model, tokenizer

def predict_and_evaluate(model, tokenizer, texts, true_labels, label_mapping):
    """预测并评估模型"""
    # 将文本标签转换为数值标签
    label_to_id = {v: int(k) for k, v in label_mapping.items()}
    true_labels_ids = [label_to_id.get(label, 0) for label in true_labels]  # 使用get避免KeyError
    
    # 创建数据集
    dataset = TextDataset(texts, true_labels_ids, tokenizer, max_len=128)
    
    # 创建Trainer进行预测
    trainer = Trainer(model=model)
    
    # 使用try-except捕获可能的错误
    try:
        predictions = trainer.predict(dataset)
        # 获取预测标签
        pred_labels = np.argmax(predictions.predictions, axis=1)
    except Exception as e:
        print(f"预测过程中出错: {e}")
        # 如果预测失败，创建一个空的预测结果
        pred_labels = np.zeros_like(true_labels_ids)
    
    # 计算每个类别的指标
    report = classification_report(true_labels_ids, pred_labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    # 添加标签名称
    def map_index_to_label(x):
        if x in ['accuracy', 'macro avg', 'weighted avg']:
            return x
        try:
            return label_mapping.get(int(x), x)
        except (ValueError, TypeError):
            return x
    
    report_df['label_name'] = report_df.index.map(map_index_to_label)
    
    # 计算混淆矩阵
    cm = confusion_matrix(true_labels_ids, pred_labels)
    
    return report_df, cm, true_labels_ids, pred_labels

def analyze_class_distribution(true_labels_ids, label_mapping):
    """分析类别分布"""
    class_counts = pd.Series(true_labels_ids).value_counts().sort_index()
    class_counts.index = class_counts.index.map(lambda x: f"{x}: {label_mapping.get(x, '')}")
    
    plt.figure(figsize=(15, 8))
    class_counts.plot(kind='bar')
    plt.title('类别分布')
    plt.xlabel('类别')
    plt.ylabel('样本数量')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'output/class_distribution.png')
    plt.close()
    
    return class_counts

def plot_metrics_by_class(report_df, output_prefix):
    """绘制每个类别的指标"""
    # 过滤掉非类别行
    class_metrics = report_df[~report_df.index.isin(['accuracy', 'macro avg', 'weighted avg'])]
    
    # 按样本数量排序
    class_metrics = class_metrics.sort_values('support', ascending=False)
    
    # 绘制精确率、召回率和F1分数
    plt.figure(figsize=(15, 10))
    
    metrics = ['precision', 'recall', 'f1-score']
    for i, metric in enumerate(metrics):
        plt.subplot(3, 1, i+1)
        sns.barplot(x=class_metrics.index, y=class_metrics[metric], palette='viridis')
        plt.title(f'每个类别的{metric}')
        plt.xticks(rotation=90)
        plt.tight_layout()
    
    plt.savefig(f'output/{output_prefix}_metrics_by_class.png')
    plt.close()
    
    # 绘制少数类的指标
    minority_classes = class_metrics.tail(10)  # 取样本数量最少的10个类别
    
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics):
        plt.subplot(3, 1, i+1)
        sns.barplot(x=minority_classes.index, y=minority_classes[metric], palette='viridis')
        plt.title(f'少数类的{metric}')
        plt.xticks(rotation=90)
        plt.tight_layout()
    
    plt.savefig(f'output/{output_prefix}_minority_class_metrics.png')
    plt.close()

def plot_confusion_matrix(cm, label_mapping, output_prefix):
    """绘制混淆矩阵"""
    # 由于类别太多，可能需要绘制一个简化版的混淆矩阵
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=False, cmap='Blues')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig(f'output/{output_prefix}_confusion_matrix.png')
    plt.close()

def main():
    # 创建输出目录
    os.makedirs('output', exist_ok=True)
    
    # 默认模型名称和检查点路径
    model_name = 'chinese-macbert-base'
    checkpoint_path = 'checkpoint-4500'
    
    # 允许从命令行参数指定模型名称和检查点
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    if len(sys.argv) > 2:
        checkpoint_path = sys.argv[2]
    
    print(f"分析模型: {model_name}, 检查点: {checkpoint_path}")
    
    # 列出可用的模型和检查点
    models_dir = 'models'
    if os.path.exists(models_dir):
        print("可用的模型:")
        for model_dir in os.listdir(models_dir):
            if os.path.isdir(os.path.join(models_dir, model_dir)):
                print(f"  - {model_dir}")
                checkpoints_dir = os.path.join(models_dir, model_dir)
                if os.path.exists(checkpoints_dir):
                    checkpoints = [d for d in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, d)) and d.startswith('checkpoint')]
                    if checkpoints:
                        print(f"    检查点: {', '.join(checkpoints)}")
    
    try:
        # 加载数据和模型
        texts, labels, label_mapping, model, tokenizer = load_data_and_model(model_name, checkpoint_path)
        
        # 预测并评估
        report_df, cm, true_labels_ids, pred_labels = predict_and_evaluate(model, tokenizer, texts, labels, label_mapping)
        
        # 分析类别分布
        class_counts = analyze_class_distribution(true_labels_ids, label_mapping)
        
        # 绘制每个类别的指标
        plot_metrics_by_class(report_df, model_name)
        
        # 绘制混淆矩阵
        plot_confusion_matrix(cm, label_mapping, model_name)
        
        # 保存详细报告
        report_df.to_csv(f'output/{model_name}_classification_report.csv')
        
        # 打印少数类的性能
        minority_classes = report_df[~report_df.index.isin(['accuracy', 'macro avg', 'weighted avg'])].sort_values('support').head(10)
        print("少数类的性能指标:")
        print(minority_classes[['precision', 'recall', 'f1-score', 'support', 'label_name']])
        
        # 打印整体性能
        print("\n整体性能指标:")
        print(report_df.loc[['macro avg', 'weighted avg']][['precision', 'recall', 'f1-score']])
    
    except Exception as e:
        print(f"分析过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 

### 代码说明

# 这段代码实现了以下功能：

# 1. **加载数据和模型**：
#    - 加载原始数据和训练好的模型
#    - 加载标签映射（从之前保存的JSON文件中）

# 2. **预测和评估**：
#    - 使用模型对数据进行预测
#    - 计算每个类别的精确率、召回率和F1分数
#    - 生成混淆矩阵

# 3. **分析类别分布**：
#    - 统计每个类别的样本数量
#    - 绘制类别分布图

# 4. **可视化每个类别的指标**：
#    - 绘制每个类别的精确率、召回率和F1分数
#    - 特别关注少数类的性能

# 5. **绘制混淆矩阵**：
#    - 可视化模型的预测结果与真实标签的对比

# 通过这个分析，你可以清楚地看到模型在每个类别上的表现，特别是少数类的表现。如果发现模型在少数类上表现较差，可能需要考虑以下解决方案：

# 1. **数据增强**：对少数类进行数据增强，增加样本数量
# 2. **类别权重**：在训练时为少数类分配更高的权重
# 3. **采样技术**：使用过采样（增加少数类样本）或欠采样（减少多数类样本）
# 4. **使用更适合不平衡数据的评估指标**：如F1分数、AUC等

# 这样的分析可以帮助你更全面地了解模型的性能，避免被整体指标所误导。
