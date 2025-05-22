import pandas as pd

data_file = 'rl_model_train_data_for_torch_1011_cleaned.txt' 
new_file = 'rl_model_train_data_for_torch_1011_cleaned_mini.txt'


df = pd.read_csv(
    data_file,
    sep='\x01',
    header=None,
    na_values='\\N',
    nrows=1000  # 只读取前1000行
)

df.to_csv(
    new_file,
    sep='\x01',
    index=False,    # 不保存行索引
    header=False,   # 不保存列标题
    na_rep='\\N'     # 将缺失值表示为 '\\N'
)

print(f"前1000行已成功保存到 {new_file}")

