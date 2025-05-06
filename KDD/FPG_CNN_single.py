import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
import random
from sklearn.metrics import accuracy_score, recall_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from difflib import SequenceMatcher


# 数据离散化函数
def discretize_pm25(series):
    bins = [0, 35, 75, 115, 150, 500]
    labels = ['优', '良', '轻度', '中度', '重度']
    return pd.cut(series, bins=bins, labels=labels, right=False)

# 序列相似度计算函数
def sequence_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


# 深度增强的CNN模型定义
class Net(nn.Module):
    def __init__(self, num_classes=5):
        super(Net, self).__init__()
        # CNN部分
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3)
        )

        # LSTM部分
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2,
                            batch_first=True, dropout=0.3)

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1))

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes))

    def forward(self, x):
        # CNN处理
        x = self.cnn(x)  # [batch, channels, seq_len]
        x = x.permute(0, 2, 1)  # [batch, seq_len, channels]

        # LSTM处理
        lstm_out, _ = self.lstm(x)

        # 注意力机制
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)

        # 分类
        output = self.classifier(context_vector)
        return output

class PM25Dataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.LongTensor([self.labels[idx]])


# 改进的预测函数 - 支持模糊匹配
def predict_next_days(previous_sequence, frequent_itemsets, cnn_model, label_encoder, n_days=2, similarity_threshold=0.6):
    predictions = []
    current_sequence = previous_sequence.copy()

    for _ in range(n_days):
        # 1. 获取CNN预测和置信度
        seq_encoded = label_encoder.transform(current_sequence[-7:])
        cnn_input = torch.FloatTensor(seq_encoded).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            cnn_output = cnn_model(cnn_input)
            cnn_probs = torch.softmax(cnn_output, dim=1).squeeze().numpy()

        # 2. 获取FP-Growth预测
        fp_probs = np.zeros(len(label_encoder.classes_))
        for _, row in frequent_itemsets.iterrows():
            itemset = tuple(row['itemsets'])
            if len(itemset) <= len(current_sequence):
                continue

            # 计算序列相似度
            similarity = sequence_similarity(current_sequence[-len(itemset) + 1:], itemset[:-1])
            if similarity > similarity_threshold:
                next_item_idx = label_encoder.transform([itemset[-1]])[0]
                fp_probs[next_item_idx] += row['support'] * similarity

        # 3. 动态加权融合
        if np.sum(fp_probs) > 0:
            fp_probs = fp_probs / np.sum(fp_probs)
            # 动态权重：基于FP-Growth预测的确定性
            fp_weight = np.max(fp_probs) * 0.7  # 可调整系数
            combined_probs = fp_weight * fp_probs + (1 - fp_weight) * cnn_probs
        else:
            combined_probs = cnn_probs

        # 4. 选择预测结果
        predicted_idx = np.argmax(combined_probs)
        predicted = label_encoder.inverse_transform([predicted_idx])[0]
        predictions.append(predicted)
        current_sequence.append(predicted)

    return predictions


# 加载训练数据
df = pd.read_csv('./train/train.csv')
df['utc_time'] = pd.to_datetime(df['utc_time'])
df['PM2.5_level'] = discretize_pm25(df['PM2.5'])

# 按站点和日期聚合
daily_df = df.groupby(['stationId', df['utc_time'].dt.date])['PM2.5_level'].agg(
    lambda x: x.mode()[0] if not x.mode().empty else np.nan
).dropna().reset_index()

# 标签编码
label_encoder = LabelEncoder()
daily_df['PM2.5_encoded'] = label_encoder.fit_transform(daily_df['PM2.5_level'])

# 准备CNN训练数据 - 按站点分别处理
window_size = 7
X = []
y = []
station_ids = daily_df['stationId'].unique()

for station in tqdm(station_ids, desc="Processing stations"):
    station_data = daily_df[daily_df['stationId'] == station]
    if len(station_data) < window_size + 1:
        continue

    for i in range(len(station_data) - window_size):
        X.append(station_data['PM2.5_encoded'].iloc[i:i + window_size].values)
        y.append(station_data['PM2.5_encoded'].iloc[i + window_size])

# 创建数据集和数据加载器
dataset = PM25Dataset(X, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 训练深度CNN模型
cnn_model = Net(num_classes=len(label_encoder.classes_))
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
cnn_model = cnn_model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(cnn_model.parameters(), lr=0.001, weight_decay=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

num_epochs = 100
for epoch in range(num_epochs):
    cnn_model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = cnn_model(inputs.unsqueeze(1))
        loss = criterion(outputs, labels.squeeze())
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.squeeze()).sum().item()

    epoch_acc = correct / total
    scheduler.step(epoch_loss)

    # 打印训练信息
    print(
        f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}, Acc: {epoch_acc:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

# 创建滑动窗口序列用于FP-Growth
sequences = []
for station in tqdm(station_ids, desc="Creating sequences"):
    station_data = daily_df[daily_df['stationId'] == station]
    if len(station_data) < window_size:
        continue

    for i in range(len(station_data) - window_size + 1):
        sequences.append(station_data['PM2.5_level'].iloc[i:i + window_size].tolist())

# 转换为事务格式
te = TransactionEncoder()
te_ary = te.fit(sequences).transform(sequences)
df_seq = pd.DataFrame(te_ary, columns=te.columns_)

# 使用FP-Growth找频繁序列模式 - 降低最小支持度以获取更多模式
min_support = 0.03
frequent_itemsets = fpgrowth(df_seq, min_support=min_support, use_colnames=True)

# 结果处理
frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: list(x))
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets['sequence'] = frequent_itemsets['itemsets'].apply(
    lambda x: ' → '.join([str(i) for i in x])
)

# 加载验证集
val = pd.read_csv('./validate/val.csv')
val['utc_time'] = pd.to_datetime(val['utc_time'])
val['PM2.5_level'] = discretize_pm25(val['PM2.5'])

# 验证集按站点和日期聚合
val_daily = val.groupby(['stationId', val['utc_time'].dt.date])['PM2.5_level'].agg(
    lambda x: x.mode()[0] if not x.mode().empty else np.nan
).dropna().reset_index()

# 准备验证数据
true_labels = []
pred_labels = []
history_window = 7

val_station_ids = val_daily['stationId'].unique()

for station in tqdm(val_station_ids, desc="Validating stations"):
    station_data = val_daily[val_daily['stationId'] == station]
    if len(station_data) < history_window + 2:
        continue

    for i in range(len(station_data) - history_window - 1):
        history = station_data['PM2.5_level'].iloc[i:i + history_window].tolist()
        true_future = station_data['PM2.5_level'].iloc[i + history_window:i + history_window + 2].tolist()

        predicted_future = predict_next_days(
            history, frequent_itemsets, cnn_model, label_encoder,
            n_days=2, similarity_threshold=0.9
        )

        true_labels.extend(true_future)
        pred_labels.extend(predicted_future)

# 计算评估指标
accuracy = accuracy_score(true_labels, pred_labels)
recall = recall_score(true_labels, pred_labels, average="macro")

print("\n验证集评估结果:")
print(f"Accuracy: {accuracy:.4f}")
print(f"macro Recall: {recall:.4f}")

# 输出分类报告
from sklearn.metrics import classification_report

print("\n分类报告:")
print(classification_report(true_labels, pred_labels, target_names=label_encoder.classes_))