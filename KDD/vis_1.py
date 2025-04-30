import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from calendar import month_abbr


def plot_monthly_pollutant(df, station_id, pollutants=['PM2.5', 'PM10'],
                                     colors=['green', 'royalblue'], figsize=(14, 8)):
    """
    绘制某站点多个污染物月平均浓度变化折线图(合并显示)

    参数:
        df: 包含空气质量数据的DataFrame
        station_id: 要分析的监测站ID
        pollutants: 要分析的污染物列表 (默认为PM2.5和PM10)
        colors: 各污染物对应的颜色列表
        figsize: 图像大小

    返回:
        matplotlib的figure对象
    """
    # 数据预处理
    df = df.copy()

    # 确保时间列是datetime类型
    if not pd.api.types.is_datetime64_any_dtype(df['utc_time']):
        df['utc_time'] = pd.to_datetime(df['utc_time'])

    # 筛选指定站点数据
    station_data = df[df['stationId'] == station_id].copy()

    # 提取月份
    station_data['month'] = station_data['utc_time'].dt.month

    # 设置绘图风格
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)

    # 为每种污染物绘制折线图
    for poll, color in zip(pollutants, colors):
        # 计算月平均值
        monthly_avg = station_data.groupby('month')[poll].mean().reset_index()

        # 绘制折线图
        line = sns.lineplot(
            x='month',
            y=poll,
            data=monthly_avg,
            marker='o',
            markersize=8,
            linewidth=2.5,
            color=color,
            ax=ax,
            label=poll
        )

        # 添加数据标签
        for i, row in monthly_avg.iterrows():
            ax.text(
                row['month'],
                row[poll] + (max(monthly_avg[poll]) * 0.05),  # 动态调整标签位置
                f"{row[poll]:.1f}",
                horizontalalignment='center',
                size=10,
                color=color
            )

    # 设置标题和标签
    ax.set_title(f'{station_id} 站点污染物月平均浓度变化', fontsize=16, pad=20)
    ax.set_xlabel('月份', fontsize=12)
    ax.set_ylabel('污染物浓度 (μg/m³)', fontsize=12)

    # 设置x轴刻度为月份缩写
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([month_abbr[i] for i in range(1, 13)])

    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.6)

    # 调整图例
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 自动调整布局
    plt.tight_layout()

    return fig

# 使用示例
df = pd.read_csv("./cleaned_data/cleaned_aq.csv")
# 假设df是您的空气质量DataFrame
station = "huairou"
station_id = station + '_aq'
#fig = plot_monthly_pollutant(df, station_id=station_id, pollutants=['PM2.5', 'PM10', 'NO2', 'O3'], colors=["black", 'grey', 'red', 'blue'])
#fig = plot_monthly_pollutant(df, station_id=station_id, pollutants=['SO2'], colors=['darkgreen'])
fig = plot_monthly_pollutant(df, station_id=station_id, pollutants=['CO'], colors=['orange'])

plt.legend()
#plt.show()
plt.savefig("./" + station + "_3.png")