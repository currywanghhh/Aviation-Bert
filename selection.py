with open('data/frequency_results.txt', 'r', encoding='utf-8') as file:  
    top_lines = [next(file).split(',')[0].split(':')[1].strip() for _ in range(1000)]    
  
# 写入新文件  
with open('data/简字简语-new.txt', 'w', encoding='utf-8') as output_file:  
    for word in top_lines:  
        output_file.write(word + '\n') 