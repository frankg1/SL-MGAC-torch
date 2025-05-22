import pandas as pd
import ast
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # 新增

# 读取数据
txt_file_path = 'rl_model_train_data_for_torch_1011_cleaned.txt'
data = pd.read_csv(txt_file_path, sep='\x01', header=None, na_values='\\N').fillna(0)

# 提取相关列
cur_common_slots_col = data.iloc[:, 70]
cur_common_signs_col = data.iloc[:, 71]

cur_item_slots_col = data.iloc[:, 74]
cur_item_signs_col = data.iloc[:, 75]

nxt_common_slots_col = data.iloc[:, 72]
nxt_common_signs_col = data.iloc[:, 73]

nxt_item_slots_col = data.iloc[:, 76]
nxt_item_signs_col = data.iloc[:, 77]

# 定义白名单
cur_common_whitelist = [1000, 1001, 1008, 1009, 1020, 1004, 1005, 1006, 1007, 1200, 1201, 1202]
nxt_common_whitelist = [1010, 1011, 1018, 1019, 1021, 1014, 1015, 1016, 1017, 1203, 1204, 1205]
item_whitelist = [100, 101, 102, 103, 105, 106, 107, 108, 109, 110]

# 将字符串转换为列表
def parse_list(s):
    if s == 0 or s == '0':
        return []
    return ast.literal_eval(s)

# 转换列
print("正在解析列数据...")
cur_common_slots = cur_common_slots_col.apply(parse_list)
cur_common_signs = cur_common_signs_col.apply(parse_list)

cur_item_slots = cur_item_slots_col.apply(parse_list)
cur_item_signs = cur_item_signs_col.apply(parse_list)

nxt_common_slots = nxt_common_slots_col.apply(parse_list)
nxt_common_signs = nxt_common_signs_col.apply(parse_list)

nxt_item_slots = nxt_item_slots_col.apply(parse_list)
nxt_item_signs = nxt_item_signs_col.apply(parse_list)

# 合并所有slots和signs
all_slots = []
all_signs = []

# 获取总行数
total_rows = len(data)

print("开始处理数据并统计频率...")

# 使用tqdm添加进度条
for i in tqdm(range(total_rows), desc="Processing Rows"):
    # 处理当前common
    slots = cur_common_slots.iloc[i]
    signs = cur_common_signs.iloc[i]
    for slot, sign in zip(slots, signs):
        if slot in cur_common_whitelist:
            all_slots.append(slot)
            all_signs.append(sign)
    # 处理下一步common
    slots = nxt_common_slots.iloc[i]
    signs = nxt_common_signs.iloc[i]
    for slot, sign in zip(slots, signs):
        if slot in nxt_common_whitelist:
            all_slots.append(slot)
            all_signs.append(sign)
    # 处理当前item
    slots = cur_item_slots.iloc[i]
    signs = cur_item_signs.iloc[i]
    for slot, sign in zip(slots, signs):
        if slot in item_whitelist:
            all_slots.append(slot)
            all_signs.append(sign)
    # 处理下一步item
    slots = nxt_item_slots.iloc[i]
    signs = nxt_item_signs.iloc[i]
    for slot, sign in zip(slots, signs):
        if slot in item_whitelist:
            all_slots.append(slot)
            all_signs.append(sign)

# 统计频率
print("统计频率...")
sign_counter = Counter(all_signs)
slot_counter = Counter(all_slots)

# 提取出现频率最高的前50个sign
top_n = 50
most_common_signs = sign_counter.most_common(top_n)
signs, counts = zip(*most_common_signs)

# 绘制sign频率直方图并保存
plt.figure(figsize=(12, 6))
sns.barplot(x=list(range(top_n)), y=counts)
plt.xlabel('Sign排名')
plt.ylabel('出现频率')
plt.title(f'最常见的前{top_n}个Sign频率分布')
plt.savefig('top_signs_frequency.png')
plt.close()

# 绘制sign频率的概率密度图并保存
plt.figure(figsize=(12, 6))
sns.kdeplot(list(sign_counter.values()), shade=True)
plt.xlabel('Sign出现次数')
plt.ylabel('概率密度')
plt.title('Sign频率的概率密度分布')
plt.savefig('signs_density_plot.png')
plt.close()

# 绘制sign频率的累积分布函数（CDF）并保存
sorted_counts = sorted(sign_counter.values())
cdf = [sum(sorted_counts[:i+1])/sum(sorted_counts) for i in range(len(sorted_counts))]
plt.figure(figsize=(12, 6))
plt.plot(sorted_counts, cdf)
plt.xlabel('Sign出现次数')
plt.ylabel('累积概率')
plt.title('Sign频率的累积分布函数')
plt.savefig('signs_cdf_plot.png')
plt.close()

# 绘制log-log图并保存
plt.figure(figsize=(12, 6))
plt.loglog(sorted_counts[::-1], range(1, len(sorted_counts)+1))
plt.xlabel('Sign出现次数（log）')
plt.ylabel('排名（log）')
plt.title('Sign频率的log-log分布')
plt.savefig('signs_loglog_plot.png')
plt.close()

# 输出总的过滤后sign数量
total_filtered_signs = sum(sign_counter.values())
print(f"\n过滤后Sign的总数：{total_filtered_signs}")

