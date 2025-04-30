import pandas as pd
import numpy as np
from typing import List, Tuple
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

def get_time_averaged_features(
        df: pd.DataFrame,
        feature_pair: Tuple[str, str] = ('PM2.5', 'PM10'),
        time_granularity: str = 'hour'
) -> List[pd.DataFrame]:
    """
    获取按时间粒度平均后的两个特征序列

    Parameters:
        df: 输入DataFrame (需包含utc_time列)
        feature_pair: 要分析的两个特征名称的元组
        time_granularity: 时间粒度 ('hour','day','week','month')

    Returns:
        包含两个特征DataFrame的列表 [feature1_df, feature2_df]
        每个DataFrame包含: time_group, station_id, feature_value
    """
    # 数据预处理
    df = df.copy()

    # 重命名列（处理可能的拼写错误）
    df = df.rename(columns={
        'ute_time': 'utc_time',
        'wind_diree': 'wind_direction'
    })

    # 确保时间列是datetime类型
    if not pd.api.types.is_datetime64_any_dtype(df['utc_time']):
        try:
            df['utc_time'] = pd.to_datetime(df['utc_time'], errors='coerce')
        except:
            raise ValueError("无法解析时间列，请检查utc_time格式")

    # 验证特征列存在
    valid_features = ['temperature', 'pressure', 'humidity', 'wind_direction',
                      'wind_speed', 'weather', 'PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']

    for feature in feature_pair:
        if feature not in df.columns:
            if feature in valid_features:
                raise ValueError(f"特征列 '{feature}' 存在于数据但未正确命名")
            else:
                raise ValueError(f"无效特征 '{feature}'，可选特征: {valid_features}")

    # 分离风速和天气（如果合并在一起）
    if 'wind_speed weather' in df.columns:
        df[['wind_speed', 'weather']] = df['wind_speed weather'].str.extract(r'(\d+\.?\d*)\s+(.*)')
        df['wind_speed'] = df['wind_speed'].astype(float)

    # 按时间粒度分组
    if time_granularity == 'hour':
        df['time_group'] = df['utc_time'].dt.hour
        time_labels = [f'Hour_{i:02d}' for i in range(24)]
    elif time_granularity == 'day':
        df['time_group'] = df['utc_time'].dt.date
        time_labels = sorted(df['time_group'].unique())
    elif time_granularity == 'week':
        df['time_group'] = df['utc_time'].dt.to_period('W').astype(str)
        time_labels = sorted(df['time_group'].unique())
    elif time_granularity == 'month':
        df['time_group'] = df['utc_time'].dt.to_period('M').astype(str)
        time_labels = sorted(df['time_group'].unique())
    else:
        raise ValueError("time_granularity 必须是 'hour', 'day', 'week' 或 'month'")

    # 计算时间粒度平均值（保留站点信息）
    grouped = df.groupby(['time_group'])[list(feature_pair)].mean().reset_index()

    # 转换为两个特征DataFrame
    result = []
    for feature in feature_pair:
        feature_df = grouped[['time_group', feature]].copy()
        feature_df.columns = ['time_group', 'value']
        feature_df['feature'] = feature
        result.append(feature_df)

    # 按时间排序
    result[0] = result[0].sort_values('time_group')['value']
    result[1] = result[1].sort_values('time_group')['value']

    return result


# 使用示例
if __name__ == "__main__":
    # 模拟数据 (替换为实际数据加载)
    srcc_matrix = {}
    df = pd.read_csv("./merged_data/miyun.csv")
    for weather_data in ['temperature', 'pressure', 'humidity', 'wind_speed', 'wind_direction']:
        for air_data in ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']:
            f1, f2 = get_time_averaged_features(
                df=df,
                feature_pair=(weather_data, air_data),
                time_granularity='day'
            )
            srcc = spearmanr(f1, f2)[0]
            srcc_matrix[weather_data, air_data] = srcc
print(srcc_matrix)
variables = sorted(list(set([x[0] for x in srcc_matrix.keys()] + [x[1] for x in srcc_matrix.keys()])))

weather_vars = sorted(list(set([k[0] for k in srcc_matrix])))
air_vars = sorted(list(set([k[1] for k in srcc_matrix])))

# 构建 DataFrame
corr_df = pd.DataFrame(index=weather_vars, columns=air_vars)
for (w, a), val in srcc_matrix.items():
    corr_df.loc[w, a] = val

# 确保所有数据是 float 类型（避免 TypeError）
print(corr_df)
corr_df = corr_df.astype(float)  # 关键修复！

# 绘制热力图
plt.figure(figsize=(10, 6))
sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, linewidths=0.5)

plt.tight_layout()
plt.savefig("./corr_miyun.png")
#plt.show()
