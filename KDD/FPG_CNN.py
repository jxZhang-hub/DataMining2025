import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
import random
from sklearn.metrics import accuracy_score, recall_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
from difflib import SequenceMatcher
import pickle


class PM25Predictor:
    def __init__(self, config):
        """初始化预测系统"""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_encoder = LabelEncoder()
        self.feature_scaler = StandardScaler()

    def discretize_pm25(self, series):
        """PM2.5离散化"""
        bins = [0, 50, 100, 150, 200, 500]
        labels = ['优', '良', '轻度', '中度', '重度']
        return pd.cut(series, bins=bins, labels=labels, right=False)

    def sequence_similarity(self, a, b):
        """序列相似度计算"""
        return SequenceMatcher(None, a, b).ratio()

    def prepare_data(self, df, is_train=True):
        """数据预处理"""
        df['utc_time'] = pd.to_datetime(df['utc_time'])
        df['PM2.5_level'] = self.discretize_pm25(df['PM2.5'])

        # 按站点和日期聚合
        daily_df = df.groupby(['station_id', df['utc_time'].dt.date]).agg({
            'PM2.5_level': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
            **{f: 'mean' for f in self.config['aux_features']}
        }).dropna().reset_index()

        # 标签编码
        if is_train:
            daily_df['PM2.5_encoded'] = self.label_encoder.fit_transform(daily_df['PM2.5_level'])
        else:
            # 确保标签编码器已经fit过
            if len(self.label_encoder.classes_) == 0:
                raise ValueError("LabelEncoder has not been fitted. Train the model first or load a trained model.")
            daily_df['PM2.5_encoded'] = self.label_encoder.transform(daily_df['PM2.5_level'])

        # 标准化辅助特征
        if is_train:
            aux_values = self.feature_scaler.fit_transform(daily_df[self.config['aux_features']])
        else:
            # 确保标准化器已经fit过
            if not hasattr(self.feature_scaler, 'mean_'):
                raise ValueError("StandardScaler has not been fitted. Train the model first or load a trained model.")
            aux_values = self.feature_scaler.transform(
                daily_df[self.config['aux_features']])  # 关键修改：使用transform而不是fit_transform

        return daily_df, aux_values

    def create_sequences(self, daily_df, aux_values):
        """创建时间序列数据"""
        X_seq, X_aux, y = [], [], []
        station_ids = daily_df['station_id'].unique()

        for station in tqdm(station_ids, desc="Processing stations"):
            station_data = daily_df[daily_df['station_id'] == station]
            if len(station_data) < self.config['window_size'] + 1:
                continue

            station_aux = aux_values[daily_df['station_id'] == station]

            for i in range(len(station_data) - self.config['window_size']):
                X_seq.append(station_data['PM2.5_encoded'].iloc[i:i + self.config['window_size']].values)
                X_aux.append(station_aux[i:i + self.config['window_size']])
                y.append(station_data['PM2.5_encoded'].iloc[i + self.config['window_size']])

        return X_seq, X_aux, y

    def train(self, train_df):
        """训练模型"""
        # 数据准备
        daily_df, aux_values = self.prepare_data(train_df, is_train=True)
        X_seq, X_aux, y = self.create_sequences(daily_df, aux_values)

        # 创建数据加载器
        dataset = PM25Dataset(X_seq, X_aux, y)
        dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)

        # 初始化模型
        self.model = MultiFeatureNet(
            num_features=len(self.config['aux_features']) + 1,
            num_classes=len(self.label_encoder.classes_)
        ).to(self.device)

        # 训练配置
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min',
            patience=self.config['lr_patience'],
            factor=self.config['lr_factor']
        )

        # 训练循环
        for epoch in range(self.config['num_epochs']):
            self.model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for seq, aux, labels in dataloader:
                seq, aux, labels = seq.to(self.device), aux.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(seq.unsqueeze(1), aux[:, -1, :])
                loss = criterion(outputs, labels.squeeze())
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.squeeze()).sum().item()

            epoch_acc = correct / total
            scheduler.step(epoch_loss)

            if (epoch + 1) % 10 == 0:
                print(
                    f'Epoch {epoch + 1}/{self.config["num_epochs"]}, Loss: {epoch_loss / len(dataloader):.4f}, Acc: {epoch_acc:.4f}')

        # 训练FP-Growth
        self.train_fpgrowth(daily_df)

    def train_fpgrowth(self, daily_df):
        """训练FP-Growth模型"""
        sequences = []
        station_ids = daily_df['station_id'].unique()

        for station in tqdm(station_ids, desc="Creating sequences"):
            station_data = daily_df[daily_df['station_id'] == station]
            if len(station_data) < self.config['window_size']:
                continue

            for i in range(len(station_data) - self.config['window_size'] + 1):
                sequences.append(station_data['PM2.5_level'].iloc[i:i + self.config['window_size']].tolist())

        te = TransactionEncoder()
        te_ary = te.fit(sequences).transform(sequences)
        df_seq = pd.DataFrame(te_ary, columns=te.columns_)

        self.frequent_itemsets = fpgrowth(df_seq, min_support=self.config['min_support'], use_colnames=True)
        self.frequent_itemsets['itemsets'] = self.frequent_itemsets['itemsets'].apply(lambda x: list(x))
        self.frequent_itemsets['length'] = self.frequent_itemsets['itemsets'].apply(lambda x: len(x))

    def evaluate(self, val_df):
        """评估模型"""
        daily_df, aux_values = self.prepare_data(val_df, is_train=False)

        true_labels, pred_labels = [], []
        val_station_ids = daily_df['station_id'].unique()

        for station in tqdm(val_station_ids, desc="Validating stations"):
            station_data = daily_df[daily_df['station_id'] == station]
            station_aux = aux_values[daily_df['station_id'] == station]

            if len(station_data) < self.config['history_window'] + 2:
                continue

            for i in range(len(station_data) - self.config['history_window'] - 1):
                history = station_data['PM2.5_level'].iloc[i:i + self.config['history_window']].tolist()
                history_aux = station_aux[i:i + self.config['history_window']]
                true_future = station_data['PM2.5_level'].iloc[
                              i + self.config['history_window']:i + self.config['history_window'] + 2].tolist()

                predicted_future = self.predict_next_days(
                    history, history_aux,
                    n_days=2,
                    similarity_threshold=self.config['similarity_threshold']
                )

                true_labels.extend(true_future)
                pred_labels.extend(predicted_future)

        # 计算指标
        accuracy = accuracy_score(true_labels, pred_labels)
        recall = recall_score(true_labels, pred_labels, average="macro")

        print("\n验证集评估结果:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"macro Recall: {recall:.4f}")
        print("\n分类报告:")
        print(classification_report(true_labels, pred_labels, target_names=self.label_encoder.classes_))

        return accuracy, recall

    def predict_next_days(self, previous_sequence, previous_features, n_days=2, similarity_threshold=0.6):
        """预测未来n天"""
        predictions = []
        current_sequence = previous_sequence.copy()
        current_features = previous_features.copy()

        for _ in range(n_days):
            # 模型预测
            seq_encoded = self.label_encoder.transform(current_sequence[-self.config['window_size']:])
            seq_input = torch.FloatTensor(seq_encoded).unsqueeze(0).unsqueeze(0).to(self.device)
            feature_input = torch.FloatTensor(
                self.feature_scaler.fit_transform([current_features[-1]])
            ).to(self.device)

            with torch.no_grad():
                output = self.model(seq_input.to(self.device), feature_input.to(self.device))
                model_probs = torch.softmax(output, dim=1).squeeze().detach().cpu().numpy()

            # FP-Growth预测
            fp_probs = np.zeros(len(self.label_encoder.classes_))
            for _, row in self.frequent_itemsets.iterrows():
                itemset = tuple(row['itemsets'])
                if len(itemset) <= len(current_sequence):
                    continue

                similarity = self.sequence_similarity(current_sequence[-len(itemset) + 1:], itemset[:-1])
                if similarity > similarity_threshold:
                    next_item_idx = self.label_encoder.transform([itemset[-1]])[0]
                    fp_probs[next_item_idx] += row['support'] * similarity

            # 融合预测
            if np.sum(fp_probs) > 0:
                fp_probs = fp_probs / np.sum(fp_probs)
                fp_weight = np.max(fp_probs) * similarity_threshold * 0.5
                combined_probs = fp_weight * fp_probs + (1 - fp_weight) * model_probs
            else:
                combined_probs = model_probs

            # 选择结果
            predicted_idx = np.argmax(combined_probs)
            predicted = self.label_encoder.inverse_transform([predicted_idx])[0]
            predictions.append(predicted)

            # 更新序列
            current_sequence.append(predicted)
            current_features = np.vstack([current_features, current_features[-1]])

        return predictions

    def save_model(self, cnn_model_path, fpgrowth_path, label_encoder_path):
        """
        保存CNN模型、FP-Growth子序列和LabelEncoder的classes_
        :param cnn_model_path: CNN模型保存路径
        :param fpgrowth_path: FP-Growth子序列保存路径
        :param label_encoder_path: LabelEncoder的classes_保存路径
        """
        torch.save(self.model.state_dict(), cnn_model_path)  # 保存CNN模型状态字典
        with open(fpgrowth_path, 'wb') as f:
            pickle.dump(self.frequent_itemsets, f)  # 保存FP-Growth子序列
        with open(label_encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder.classes_, f)  # 保存LabelEncoder的classes_

    def load_model(self, cnn_model_path, fpgrowth_path, label_encoder_path):
        """
        加载CNN模型、FP-Growth子序列和LabelEncoder的classes_
        :param cnn_model_path: CNN模型加载路径
        :param fpgrowth_path: FP-Growth子序列加载路径
        :param label_encoder_path: LabelEncoder的classes_加载路径
        """
        # 首先加载LabelEncoder的classes_
        with open(label_encoder_path, 'rb') as f:
            classes = pickle.load(f)
            self.label_encoder.classes_ = classes

        # 然后初始化并加载模型
        self.model = MultiFeatureNet(
            num_features=len(self.config['aux_features']) + 1,
            num_classes=len(self.label_encoder.classes_))
        self.model = self.model.to(self.device)

        # 加载模型状态字典
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(cnn_model_path))
        else:
            self.model.load_state_dict(torch.load(cnn_model_path, map_location=torch.device('cpu')))

        self.model.eval() # 设置模型为评估模式

        # 最后加载FP-Growth子序列
        with open(fpgrowth_path, 'rb') as f:
            self.frequent_itemsets = pickle.load(f)


class PM25Dataset(Dataset):
    """自定义数据集类"""

    def __init__(self, sequences, aux_features, labels):
        self.sequences = sequences
        self.aux_features = aux_features
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (torch.FloatTensor(self.sequences[idx]),
                torch.FloatTensor(self.aux_features[idx]),
                torch.LongTensor([self.labels[idx]]))


class MultiFeatureNet(nn.Module):
    """多特征神经网络模型"""

    def __init__(self, num_features, num_classes=5):
        super(MultiFeatureNet, self).__init__()
        # 辅助特征分支
        self.aux_feature = nn.Sequential(
            nn.Linear(num_features - 1, 64),
            nn.ReLU(),
            nn.Dropout(0.3))

        # 时间序列分支
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3))

        self.lstm = nn.LSTM(128, 256, num_layers=2, batch_first=True, dropout=0.3)
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1))

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes))

    def forward(self, x_seq, x_aux):
        aux_features = self.aux_feature(x_aux)

        x_seq = self.cnn(x_seq)
        x_seq = x_seq.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x_seq)

        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        seq_features = torch.sum(attention_weights * lstm_out, dim=1)

        combined = torch.cat([seq_features, aux_features], dim=1)
        return self.fusion(combined)


# 使用示例
if __name__ == "__main__":
    # 配置参数
    config = {
        'aux_features': ['temperature', 'humidity', 'wind_speed', 'pressure'],
        'window_size': 7,
        'history_window': 7,
        'batch_size': 64,
        'num_epochs': 200,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'lr_patience': 5,
        'lr_factor': 0.5,
        'min_support': 0.01,
        'similarity_threshold': 0.3
    }

    # 初始化预测器
    predictor = PM25Predictor(config)

    # 加载数据
    train_df = pd.read_csv('./train/train_.csv')
    val_df = pd.read_csv('./validate/val_.csv')

    # 训练模型
    predictor.train(train_df)
    # 保存模型
    model_path = 'model.pth'
    fpgrowth_path = 'fpgrowth.pkl'
    label_encoder_path = 'label_encoder_classes.pkl'
    predictor.save_model(model_path, fpgrowth_path, label_encoder_path)

    # 加载模型
    predictor.load_model(model_path, fpgrowth_path, label_encoder_path)

    # 评估模型
    predictor.evaluate(val_df)

    # 示例预测
    sample_history = ['良', '良', '良', '良', '良', '中度', '中度']
    sample_features = [[22.45, 13.2, 0, 1010], [23.95, 37, 0, 1010], [26.15, 53.6, 1, 1009],
                       [26.05, 22.2, 0, 1010], [25.75, 37.5, 0, 1011], [31.2, 38.5, 0, 1011],
                       [27.4, 33.8, 0, 1010]]
    predictions = predictor.predict_next_days(sample_history, sample_features, n_days=2)
    print(f"\n示例预测结果: {predictions}")