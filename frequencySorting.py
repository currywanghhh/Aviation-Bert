import jieba  
from collections import Counter  
  
# 读取词汇表  
with open('data/简字简语.txt', 'r', encoding='utf-8') as vocab_file:  
    vocab_set = set(line.strip() for line in vocab_file if line.strip())  
  
# 初始化计数器  
word_counts = Counter()  
  
# 使用jieba进行分词并统计频率  
with open('data/不安全事件分类整理.txt', 'r', encoding='utf-8') as data_file:  
    for line in data_file:  
        # 去除行尾的换行符  
        line = line.strip()  
        if not line:  
            continue  
          
        # 使用jieba进行分词  
        words = jieba.lcut(line)  
          
        # 统计出现在词汇表中的词语的频率  
        for word in words:  
            if len(word)!=1 and word in vocab_set:  
                word_counts[word] += 1    
  
# 如果需要，可以将结果写入文件  
with open('data/frequency_results.txt', 'w', encoding='utf-8') as result_file:  
    for word, freq in word_counts.most_common():  
        result_file.write(f"Word: {word}, Frequency: {freq}\n")

