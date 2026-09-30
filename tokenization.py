from transformers import BertTokenizer, BertForPreTraining

# 定义词汇表文件路径
vocab_file = 'data/简字简语-new.txt'

# 读取自定义词汇表
with open(vocab_file, 'r', encoding='utf-8') as f:
    new_tokens = f.read().splitlines()

# 加载现有的 bert-base-chinese 分词器
tokenizer = BertTokenizer.from_pretrained('pretrained-models/bert-base-chinese')

# 获取原始词汇表
original_vocab_size = len(tokenizer)

# 添加自定义词汇到词汇表
new_token_ids = {}
for token in new_tokens:
    if token not in tokenizer.get_vocab():
        tokenizer.add_tokens(token)
        new_token_ids[token] = tokenizer.convert_tokens_to_ids(token)

# 更新模型词汇表
model = BertForPreTraining.from_pretrained('pretrained-models/bert-base-chinese')
model.resize_token_embeddings(len(tokenizer))

# 保存新的分词器和模型
tokenizer.save_pretrained('pretrained-models/update-toke-bert-base-chinese')
model.save_pretrained('pretrained-models/update-toke-bert-base-chinese')

