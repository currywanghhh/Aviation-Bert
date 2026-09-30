from flask import Flask, render_template_string, request
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import io
import base64

app = Flask(__name__)

# 加载模型和分词器
tokenizer = BertTokenizer.from_pretrained('pretrained-models/update-toke-bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('models/bert-classification')

# 加载标签映射
with open('data/label_mapping.json', 'r', encoding='utf-8') as f:
    label_mapping = json.load(f)
label_options = list(label_mapping.values())

# 手动指定字体路径
font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
font_prop = fm.FontProperties(fname=font_path)

# 设置字体属性
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def predict(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    top5_probs, top5_indices = torch.topk(probs, k=5, dim=-1)
    top5_predictions = [
        {"label": f"Label{i + 1}", "full_label": label_mapping.get(str(label_id), "Unknown"), "prob": prob}
        for i, (label_id, prob) in enumerate(zip(top5_indices[0].tolist(), top5_probs[0].tolist()))
    ]
    return top5_predictions


def plot_probabilities(predictions):
    labels = [pred["label"] for pred in predictions]
    probs = [pred["prob"] for pred in predictions]

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(labels, probs, color='skyblue')
    ax.set_ylabel("概率", fontproperties=font_prop)
    ax.set_title("Top5预测结果", fontproperties=font_prop)
    ax.set_ylim(0, 1)

    for i, v in enumerate(probs):
        ax.text(i, v + 0.01, f"{v:.4f}", ha='center', color='blue', fontproperties=font_prop)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url


@app.route('/', methods=['GET', 'POST'])
def index():
    top5_predictions = []
    message = ""
    plot_url = None
    if request.method == 'POST':
        text = request.form.get('text')
        if not text:
            message = "请输入待预测的内容！"
        else:
            top5_predictions = predict(text)
            plot_url = plot_probabilities(top5_predictions)
            selected_label = request.form.get('label')
            if selected_label:
                excel_path = 'data/备注和事件分类全.xlsx'
                new_data = pd.DataFrame([{"text": text, "label": selected_label}])
                try:
                    existing_data = pd.read_excel(excel_path)
                    updated_data = pd.concat([existing_data, new_data], ignore_index=True)
                except FileNotFoundError:
                    updated_data = new_data
                updated_data.to_excel(excel_path, index=False)
                message = "保存成功"

    return render_template_string('''
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <title>航空告警分类识别</title>
            <style>
                body {
                    font-family: 'Microsoft YaHei', 'SimSun', 'SimHei', Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    display: flex;
                    background-color: #f4f4f9;
                    color: #333;
                }
                .container {
                    width: 100%;
                    max-width: 1200px;
                    margin: 0 auto;
                    display: flex;
                    flex-wrap: wrap;
                    padding: 20px;
                    box-sizing: border-box;
                }
                .left-section, .right-section {
                    width: 50%;
                    padding: 20px;
                    box-sizing: border-box;
                }
                .left-section {
                    background-color: #fff;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    border-radius: 8px;
                }
                .right-section {
                    background-color: #e9ecef;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    border-radius: 8px;
                    border-left: 1px solid #ccc;
                }
                textarea {
                    width: 100%;
                    height: 200px;
                    margin-bottom: 10px;
                    padding: 10px;
                    border-radius: 4px;
                    border: 1px solid #ccc;
                    font-size: 16px;
                    box-sizing: border-box;
                }
                h1, h2 {
                    margin-top: 0;
                    color: #007bff;
                }
                img {
                    max-width: 100%;
                    height: auto;
                    margin-top: 20px;
                    border-radius: 8px;
                }
                input[type="submit"] {
                    background-color: #007bff;
                    color: #fff;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 16px;
                }
                input[type="submit"]:hover {
                    background-color: #0056b3;
                }
                select {
                    padding: 5px;
                    border-radius: 4px;
                    border: 1px solid #ccc;
                    font-size: 16px;
                }
                ul {
                    list-style-type: none;
                    padding: 0;
                }
                li {
                    padding: 5px 0;
                }
                a {
                    color: #007bff;
                    text-decoration: none;
                }
                a:hover {
                    text-decoration: underline;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="left-section">
                    <h1>请输入待预测文本</h1>
                    {% if message %}
                        <p>{{ message }}</p>
                    {% endif %}
                    <form method="post">
                        <textarea id="text" name="text"></textarea><br>
                        <input type="submit" value="分类预测">
                    </form>
                </div>
                <div class="right-section">
                    {% if top5_predictions %}
                        <h2>预测结果：</h2>
                        <ul>
                            {% for pred in top5_predictions %}
                                <li>{{ pred.full_label }}: {{ pred.prob }}</li>
                            {% endfor %}
                        </ul>
                        <form method="post">
                            <input type="hidden" name="text" value="{{ request.form.get('text') }}">
                            <label for="label">选择标签：</label>
                            <select id="label" name="label">
                                {% for option in label_options %}
                                    <option value="{{ option }}">{{ option }}</option>
                                {% endfor %}
                            </select>
                            <input type="submit" value="保存结果">
                        </form>
                        {% if plot_url %}
                            <img src="data:image/png;base64,{{ plot_url }}" alt="Top5预测结果柱状图">
                        {% endif %}
                    {% endif %}
                    <a href="{{ url_for('download_history') }}">下载历史数据</a>
                </div>
            </div>
        </body>
        </html>
    ''', top5_predictions=top5_predictions, label_options=label_options, message=message, plot_url=plot_url)


@app.route('/download_history', methods=['GET'])
def download_history():
    excel_path = 'data/备注和事件分类全.xlsx'
    try:
        df = pd.read_excel(excel_path)
        response = app.response_class(
            response=df.to_excel(sep='\t', na_rep='nan'),
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers={'Content-Disposition': 'attachment;filename=备注和事件分类全.xlsx'}
        )
        return response
    except Exception as e:
        return f"下载失败: {e}", 500
    
    


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)