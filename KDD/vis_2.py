import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist
from tqdm import tqdm

# 站点坐标字典 (经度, 纬度)
station_coords = {
    'dongsi_aq': (116.417, 39.929),
    'tiantan_aq': (116.407, 39.886),
    'guanyuan_aq': (116.339, 39.929),
    'wanshouxigong_aq': (116.352, 39.878),
    'aotizhongxin_aq': (116.397, 39.982),
    'nongzhanguan_aq': (116.461, 39.937),
    'wanliu_aq': (116.287, 39.987),
    'beibuxinqu_aq': (116.174, 40.09),
    'zhiwuyuan_aq': (116.207, 40.002),
    'fengtaihuayuan_aq': (116.279, 39.863),
    'yungang_aq': (116.146, 39.824),
    'gucheng_aq': (116.184, 39.914),
    'fangshan_aq': (116.136, 39.742),
    'daxing_aq': (116.404, 39.718),
    'yizhuang_aq': (116.506, 39.795),
    'tongzhou_aq': (116.663, 39.886),
    'shunyi_aq': (116.655, 40.127),
    'pingchang_aq': (116.23, 40.217),
    'mentougou_aq': (116.106, 39.937),
    'pinggu_aq': (117.1, 40.143),
    'huairou_aq': (116.628, 40.328),
    'miyun_aq': (116.832, 40.37),
    'yanqin_aq': (115.972, 40.453),
    'dingling_aq': (116.22, 40.292),
    'badaling_aq': (115.988, 40.365),
    'miyunshuiku_aq': (116.911, 40.499),
    'donggaocun_aq': (117.12, 40.1),
    'yongledian_aq': (116.783, 39.712),
    'yufa_aq': (116.3, 39.52),
    'liulihe_aq': (116.0, 39.58),
    'qianmen_aq': (116.395, 39.899),
    'yongdingmennei_aq': (116.394, 39.876),
    'xizhimenbei_aq': (116.349, 39.954),
    'nansanhuan_aq': (116.368, 39.856),
    'dongsihuan_aq': (116.483, 39.939)
}


def get_pollutant_series(
        df: pd.DataFrame,
        station_id: str,
        pollutant: str = 'PM2.5',
        time_granularity: str = 'month',
) -> list:
    """
    Get time-based average series of a pollutant for a specific station

    Parameters:
        df: DataFrame containing air quality data
        station_id: Monitoring station ID
        pollutant: Pollutant name (e.g., 'PM2.5')
        time_granularity: Time granularity, options:
            - 'hour'       : Hourly (24 hours)
            - 'day'        : Daily (365 days)
            - 'week'       : Weekly (52 weeks)
            - 'month'      : Monthly (12 months)
            - 'hourly_year': Yearly hourly (365×24=8760 hours)

    Returns:
        List of average values with length corresponding to time granularity
    """
    # Data preprocessing
    df = df.copy()
    df['utc_time'] = pd.to_datetime(df['utc_time'])
    station_data = df[df['stationId'] == station_id].copy()

    # Group by time granularity
    if time_granularity == 'hour':
        # Group by hour of day (24 hours)
        station_data['time_group'] = station_data['utc_time'].dt.hour
        n = 24
    elif time_granularity == 'day':
        # Group by day of year (365 days)
        station_data['time_group'] = station_data['utc_time'].dt.dayofyear
        n = 365
    elif time_granularity == 'week':
        # Group by week of year (ISO week)
        station_data['time_group'] = station_data['utc_time'].dt.isocalendar().week
        n = 52
    elif time_granularity == 'month':
        # Group by month (12 months)
        station_data['time_group'] = station_data['utc_time'].dt.month
        n = 12
    elif time_granularity == 'hourly_year':
        # Group by hour of year (8760 hours)
        station_data['time_group'] = (
                (station_data['utc_time'].dt.dayofyear - 1) * 24  # Day of year (0-364)
                + station_data['utc_time'].dt.hour  # Hour (0-23)
        )
        n = 365 * 24  # 8760
    else:
        raise ValueError(
            "time_granularity must be: 'hour', 'day', 'week', 'month' or 'hourly_year'"
        )

    # Calculate averages and fill complete series
    avg_series = station_data.groupby('time_group')[pollutant].mean()
    result = [avg_series.get(i, float('nan')) for i in range(n)]

    return result

def get_sorted_stations(station_list):
    """
    根据地理坐标对站点进行空间排序
    Parameters:
        station_list: list of station IDs
    Returns:
        sorted_stations: 按空间分布排序后的站点列表
        dist_matrix: 距离矩阵 (km)
    """
    # 提取坐标并计算距离矩阵
    coords = np.array([station_coords[s] for s in station_list])
    dist_matrix = cdist(coords, coords, metric='euclidean') * 111  # 1度≈111km

    # 使用第一个站点作为起点
    sorted_idx = [0]
    remaining_idx = set(range(1, len(station_list)))

    # 最近邻排序算法
    while remaining_idx:
        last = sorted_idx[-1]
        next_idx = min(remaining_idx, key=lambda x: dist_matrix[last, x])
        sorted_idx.append(next_idx)
        remaining_idx.remove(next_idx)

    sorted_stations = [station_list[i] for i in sorted_idx]
    return sorted_stations, dist_matrix

def plot_station_srcc_heatmap(
        df: pd.DataFrame,
        station_category: str = "Urban",
        pollutant: str = "PM2.5",
        time_granularity: str = "hourly_year",
        figsize: tuple = (12, 10)
) -> None:
    """
    绘制基于空间排序的SRCC热力图（直接使用stationId作为标签）
    Parameters:
        df: 空气质量DataFrame
        station_category: 站点类别 ("Urban"/"Suburban"/"Background"/"Traffic")
        pollutant: 污染物类型
        time_granularity: 时间粒度
        figsize: 图像尺寸
    """
    # 1. 定义站点分类
    station_categories = {
        "Urban": ['dongsi_aq', 'tiantan_aq', 'guanyuan_aq', 'wanshouxigong_aq',
                 'aotizhongxin_aq', 'nongzhanguan_aq', 'wanliu_aq', 'beibuxinqu_aq',
                 'zhiwuyuan_aq', 'fengtaihuayuan_aq', 'yungang_aq', 'gucheng_aq'],
        "Suburban": ['fangshan_aq', 'daxing_aq', 'yizhuang_aq', 'tongzhou_aq',
                    'shunyi_aq', 'pingchang_aq', 'mentougou_aq', 'pinggu_aq',
                    'huairou_aq', 'miyun_aq', 'yanqin_aq'],
        "Background": ['dingling_aq', 'badaling_aq', 'miyunshuiku_aq', 'donggaocun_aq',
                      'yongledian_aq', 'yufa_aq', 'liulihe_aq'],
        "Traffic": ['qianmen_aq', 'yongdingmennei_aq', 'xizhimenbei_aq',
                   'nansanhuan_aq', 'dongsihuan_aq']
    }

    if station_category not in station_categories:
        raise ValueError(f"station_category must be: {list(station_categories.keys())}")

    target_stations = station_categories[station_category]

    # 2. 按空间位置排序站点
    sorted_stations, _ = get_sorted_stations(target_stations)
    print("空间排序后的站点顺序:", sorted_stations)

    # 3. 预计算各站点数据
    print(f"计算{station_category}站点{pollutant}数据...")
    station_data = {}
    for station in tqdm(sorted_stations):
        station_data[station] = get_pollutant_series(
            df, station, pollutant, time_granularity
        )

    # 4. 计算SRCC矩阵
    n_stations = len(sorted_stations)
    srcc_matrix = pd.DataFrame(
        np.nan,
        index=sorted_stations,
        columns=sorted_stations
    )

    print("计算SRCC矩阵...")
    for i in tqdm(range(n_stations)):
        for j in range(i, n_stations):  # 对称矩阵，只需计算上三角
            s1 = pd.Series(station_data[sorted_stations[i]])
            s2 = pd.Series(station_data[sorted_stations[j]])
            valid_mask = s1.notna() & s2.notna()
            if valid_mask.sum() >= 2:
                srcc = spearmanr(s1[valid_mask], s2[valid_mask]).correlation
                srcc_matrix.iloc[i, j] = srcc
                srcc_matrix.iloc[j, i] = srcc

    # 5. 绘制热力图（直接使用DataFrame的索引和列名作为标签）
    plt.figure(figsize=figsize)
    sns.heatmap(
        srcc_matrix,
        cmap="coolwarm",
        vmin=0.5,
        vmax=1,
        center=0.75,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Spearman Correlation Coefficient"}
    )

    plt.title(
        f"{station_category} Stations {pollutant} {time_granularity}\nSRCC Heatmap (Spatially Ordered)",
        pad=20,
        fontsize=12
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{station_category}_{pollutant}.png")
    #plt.show()

# 使用示例
# 计算城区各站点观测SO2的相关系数
# 按“time_granularity”计算平均
df = pd.read_csv("./cleaned_data/cleaned_aq.csv")
plot_station_srcc_heatmap(
    df,
    station_category="Urban",
    pollutant="SO2",
    time_granularity="day",
    figsize=(10, 8))