import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
import random
from sklearn.metrics import accuracy_score, recall_score

random.seed(3307)

# 数据离散化函数
def discretize_pm25(series):
    bins = [0, 35, 75, 115, 150, 500]
    labels = ['优', '良', '轻度', '中度', '重度']
    return pd.cut(series, bins=bins, labels=labels, right=False)


# 支持度加权预测
def predict_next_days(previous_sequence, frequent_itemsets, n_days=2):
    predictions = []
    current_sequence = previous_sequence.copy()

    # 将itemsets转换为可比较的元组形式
    frequent_itemsets['itemsets_tuple'] = frequent_itemsets['itemsets'].apply(lambda x: tuple(x))

    for _ in range(n_days):
        matched = None
        # 尝试从长到短匹配序列
        for length in range(min(len(current_sequence), 7), 0, -1):
            recent_subseq = tuple(current_sequence[-length:])
            matches = frequent_itemsets[frequent_itemsets['itemsets_tuple'] == recent_subseq]

            if not matches.empty:
                matched = matches.iloc[0]
                break

        if matched is None:
            # 如果没有匹配，使用支持度加权的单项
            single_items = frequent_itemsets[frequent_itemsets['length'] == 1]
            if not single_items.empty:
                items = list(single_items['itemsets'].apply(lambda x: list(x)[0]))
                weights = list(single_items['support'])
                predicted = random.choices(items, weights=weights, k=1)[0]
            else:
                # 如果连单项都没有，使用最近一天的值
                predicted = current_sequence[-1]
            predictions.append(predicted)
            current_sequence.append(predicted)
            continue

        # 查找可能的扩展
        possible_extensions = []
        for _, row in frequent_itemsets.iterrows():
            if row['length'] > len(matched['itemsets_tuple']):
                if row['itemsets_tuple'][:len(matched['itemsets_tuple'])] == matched['itemsets_tuple']:
                    next_item = row['itemsets_tuple'][len(matched['itemsets_tuple'])]
                    possible_extensions.append((next_item, row['support']))

        if possible_extensions:
            # 支持度加权随机选择
            items, weights = zip(*possible_extensions)
            predicted = random.choices(items, weights=weights, k=1)[0]
        else:
            # 如果没有扩展，使用支持度加权的单项
            single_items = frequent_itemsets[frequent_itemsets['length'] == 1]
            if not single_items.empty:
                items = list(single_items['itemsets'].apply(lambda x: list(x)[0]))
                weights = list(single_items['support'])
                predicted = random.choices(items, weights=weights, k=1)[0]
            else:
                predicted = current_sequence[-1]

        predictions.append(predicted)
        current_sequence.append(predicted)

    return predictions


# 加载训练数据
df = pd.read_csv('./train/train.csv')
df['utc_time'] = pd.to_datetime(df['utc_time'])
# 数据处理流程
df['PM2.5_level'] = discretize_pm25(df['PM2.5'])

# 按天聚合
daily_df = df.groupby(df['utc_time'].dt.date)['PM2.5_level'].agg(
    lambda x: x.mode()[0] if not x.mode().empty else np.nan
).dropna().reset_index()

# 创建滑动窗口序列
window_size = 7
sequences = []
for i in range(len(daily_df) - window_size + 1):
    window = daily_df['PM2.5_level'].iloc[i:i + window_size].tolist()
    sequences.append(window)

# 转换为事务格式
te = TransactionEncoder()
te_ary = te.fit(sequences).transform(sequences)
df_seq = pd.DataFrame(te_ary, columns=te.columns_)

# 使用FP-Growth找频繁序列模式
min_support = 0.05
frequent_itemsets = fpgrowth(df_seq, min_support=min_support, use_colnames=True)

# 结果处理
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets['sequence'] = frequent_itemsets['itemsets'].apply(
    lambda x: ' → '.join([str(i) for i in x])
)

'''
previous_sequence = ['优', '良', '中度', '良', '重度']
predictions = predict_next_days(previous_sequence, frequent_itemsets, n_days=2)
print("测试预测结果:", predictions)
'''

# 加载验证集
val = pd.read_csv('./validate/val.csv')
val['utc_time'] = pd.to_datetime(val['utc_time'])
val['PM2.5_level'] = discretize_pm25(val['PM2.5'])

# 验证集按天聚合
val_daily = val.groupby(val['utc_time'].dt.date)['PM2.5_level'].agg(
    lambda x: x.mode()[0] if not x.mode().empty else np.nan
).dropna().reset_index()

# 准备验证数据
true_labels = []
pred_labels = []
window_size = 5  # 使用5天历史预测未来2天

for i in range(len(val_daily) - window_size - 1):
    # 获取历史序列
    history = val_daily['PM2.5_level'].iloc[i:i + window_size].tolist()

    # 获取真实未来2天数据
    true_future = val_daily['PM2.5_level'].iloc[i + window_size:i + window_size + 2].tolist()

    # 预测未来2天
    predicted_future = predict_next_days(history, frequent_itemsets, n_days=2)

    # 保存结果
    true_labels.extend(true_future)
    pred_labels.extend(predicted_future)

# 计算评估指标
accuracy = accuracy_score(true_labels, pred_labels)
recall = recall_score(true_labels, pred_labels, average="macro")  # 加权平均召回率

print("\n验证集评估结果:")
print(f"Accuracy: {accuracy:.4f}")
print(f"macro Recall: {recall:.4f}")

# 输出分类报告
from sklearn.metrics import classification_report

print("\n分类报告:")
print(classification_report(true_labels, pred_labels, target_names=['优', '良', '轻度', '中度', '重度']))