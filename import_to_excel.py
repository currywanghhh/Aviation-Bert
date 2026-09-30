import pandas as pd

def import_data_to_excel(input_file, output_file):
    # 读取记事本文件并解析数据
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # 取第二列和第三列数据，并去除空格
    data = [
        [item.strip() for item in line.strip().split('|')[1:3]]
        for line in lines if '|' in line
    ]
    
    # 创建DataFrame
    df_new = pd.DataFrame(data, columns=['text', 'label'])
    
    # 读取现有的Excel文件
    df_existing = pd.read_excel(output_file)
    
    # 合并数据
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    
    # 保存合并后的数据到新的Excel文件
    df_combined.to_excel(output_file, index=False)
    print(f"Data has been successfully imported and combined into {output_file}")

# 输入文件路径和输出文件路径
input_file = 'C://Users//25497//Desktop//论文初稿//增强样本第二版7.7.txt'  # 记事本文件路径
output_file = 'C://Users//25497//Desktop//论文初稿//增强备注和事件分类全第二版7.7.xlsx'  # Excel文件路径
# 调用函数
import_data_to_excel(input_file, output_file)