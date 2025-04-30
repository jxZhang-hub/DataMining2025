import pandas as pd
from scipy import stats
import numpy as np
import json
from collections import defaultdict
import time
def load_neighbor_graph(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)

    # 转换格式确保一致性
    air_quality_graph = defaultdict(dict)
    for station, neighbors in graph_data.items():
        air_quality_graph[station] = {k: float(v) for k, v in neighbors.items()}
    return air_quality_graph


def filling_short(data, threshold=10):
    df_processed = data.copy()
    for col in df_processed.columns:
        if col != 'stationId' and col != 'utc_time':
            df_processed[col].interpolate(method='linear', inplace=True, limit=threshold)
    return df_processed

def filling_long(df, air_quality_graph, target_column='PM10'):
    """
    基于8邻域距离加权填充缺失值
    :param df: 空气质量数据DataFrame
    :param air_quality_graph: 邻域图字典 {站名: {邻居站名: 距离}}
    :param target_column: 要填充的目标列
    :return: 填充后的DataFrame
    """
    # 创建副本避免修改原数据
    filled_df = df.copy()

    # 按站点分组处理
    for station, group in filled_df.groupby('stationId'):
        if station not in air_quality_graph:
            continue  # 如果没有邻域信息则跳过

        neighbors = air_quality_graph[station]

        # 找出当前站点的缺失值索引
        missing_mask = group[target_column].isna()
        missing_indices = group[missing_mask].index

        if len(missing_indices) == 0:
            continue  # 没有缺失值则跳过

        # 对每个缺失值时间点进行填充
        for idx in missing_indices:
            # 获取当前时间点
            current_time = filled_df.at[idx, 'utc_time']

            total_weight = 0
            weighted_sum = 0

            # 检查每个邻居
            for neighbor, distance in neighbors.items():
                # 获取邻居在当前时间点的数据
                neighbor_data = filled_df[
                    (filled_df['stationId'] == neighbor) &
                    (filled_df['utc_time'] == current_time)
                    ]

                # 如果邻居有数据且不是缺失值
                if not neighbor_data.empty and not pd.isna(neighbor_data[target_column].values[0]):
                    # 计算权重 (距离越近权重越大)
                    weight = 1 / (distance ** 2)  # 使用距离平方的倒数作为权重
                    weighted_sum += neighbor_data[target_column].values[0] * weight
                    total_weight += weight

            # 如果有有效的邻居数据则进行填充
            if total_weight > 0:
                filled_value = weighted_sum / total_weight
                filled_df.at[idx, target_column] = filled_value
                #print(f"填充: 站点 {station} 在时间 {current_time} 的 {target_column} = {filled_value:.2f} (基于 {len(neighbors)} 个邻居)")
            #else:
                #print(f"警告: 站点 {station} 在时间 {current_time} 的 {target_column} 无有效邻居数据可填充")

    return filled_df


def clean_air_quality(input_file, output_file):
    data = pd.read_csv(input_file)
    print("-------------------------------------------------------------------------------")
    start = time.time()
    missing_stats = pd.DataFrame({
        '缺失值数量': data.isna().sum(),
        '缺失值比例%': data.isna().mean() * 100
    })
    end = time.time()
    print(missing_stats)
    print("用时:" + str(end - start))
    print("-------------------------------------------------------------------------------")
    start = time.time()
    data = data.dropna(thresh=3)
    #若某条监测数据缺失3个，则drop
    missing_stats = pd.DataFrame({
        '缺失值数量': data.isna().sum(),
        '缺失值比例%': data.isna().mean() * 100
    })
    end = time.time()
    print(missing_stats)
    print("用时:" + str(end - start))
    print("-------------------------------------------------------------------------------")
    print("填充短缺失")
    start = time.time()
    data = filling_short(data, threshold=24)
    missing_stats = pd.DataFrame({
        '缺失值数量': data.isna().sum(),
        '缺失值比例%': data.isna().mean() * 100
    })
    end = time.time()
    print(missing_stats)
    print("用时:" + str(end - start))
    print("-------------------------------------------------------------------------------")
    print("填充长缺失")
    start = time.time()
    aq_graph = load_neighbor_graph('neighbor_graph.json')
    data = filling_long(data, air_quality_graph=aq_graph, target_column='PM2.5')
    data = filling_long(data, air_quality_graph=aq_graph, target_column='PM10')
    data = filling_long(data, air_quality_graph=aq_graph, target_column='NO2')
    data = filling_long(data, air_quality_graph=aq_graph, target_column='CO')
    data = filling_long(data, air_quality_graph=aq_graph, target_column='O3')
    data = filling_long(data, air_quality_graph=aq_graph, target_column='SO2')
    missing_stats = pd.DataFrame({
        '缺失值数量': data.isna().sum(),
        '缺失值比例%': data.isna().mean() * 100
    })
    end = time.time()
    print(missing_stats)
    print("用时:" + str(end - start))
    data.to_csv(output_file, sep=',', index=False)

def final_clean(input_file, output_file):
    data = pd.read_csv(input_file)
    start = time.time()
    data = data.dropna()
    # 若某条监测数据缺失3个，则drop
    missing_stats = pd.DataFrame({
        '缺失值数量': data.isna().sum(),
        '缺失值比例%': data.isna().mean() * 100
    })
    end = time.time()
    print(missing_stats)
    print("用时:" + str(end - start))

    data.to_csv(output_file, sep=',', index=False)

# 使用示例
if __name__ == "__main__":
    input_file = "./beijing_17_18_aq.csv"  # 替换为您的输入文件路径
    output_file = "cleaned_aq.csv"  # 替换为您想要的输出文件路径
    #clean_air_quality(input_file, output_file)
    final_clean(output_file, "cleaned_aq_.csv")