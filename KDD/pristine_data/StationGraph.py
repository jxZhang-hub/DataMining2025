import math
from collections import defaultdict

# 从Excel数据中提取的监测站坐标数据
stations = {
    # 城区环境评价点12个
    "dongsi_aq": (116.417, 39.929),
    "tiantan_aq": (116.407, 39.886),
    "guanyuan_aq": (116.339, 39.929),
    "wanshouxigong_aq": (116.352, 39.878),
    "aotizhongxin_aq": (116.397, 39.982),
    "nongzhanguan_aq": (116.461, 39.937),
    "wanliu_aq": (116.287, 39.987),
    "beibuxinqu_aq": (116.174, 40.09),
    "zhiwuyuan_aq": (116.207, 40.002),
    "fengtaihuayuan_aq": (116.279, 39.863),
    "yungang_aq": (116.146, 39.824),
    "gucheng_aq": (116.184, 39.914),

    # 郊区环境评价点11个
    "fangshan_aq": (116.136, 39.742),
    "daxing_aq": (116.404, 39.718),
    "yizhuang_aq": (116.506, 39.795),
    "tongzhou_aq": (116.663, 39.886),
    "shunyi_aq": (116.655, 40.127),
    "pingchang_aq": (116.23, 40.217),
    "mentougou_aq": (116.106, 39.937),
    "pinggu_aq": (117.1, 40.143),
    "huairou_aq": (116.628, 40.328),
    "miyun_aq": (116.832, 40.37),
    "yanqin_aq": (115.972, 40.453),

    # 对照点及区域点7个
    "dingling_aq": (116.22, 40.292),
    "badaling_aq": (115.988, 40.365),
    "miyunshuiku_aq": (116.911, 40.499),
    "donggaocun_aq": (117.12, 40.1),
    "yongledian_aq": (116.783, 39.712),
    "yufa_aq": (116.3, 39.52),
    "liulihe_aq": (116.0, 39.58),

    # 交通污染监控点5个
    "qianmen_aq": (116.395, 39.899),
    "yongdingmennei_aq": (116.394, 39.876),
    "xizhimenbei_aq": (116.349, 39.954),
    "nansanhuan_aq": (116.368, 39.856),
    "dongsihuan_aq": (116.483, 39.939)
}


def haversine(lon1, lat1, lon2, lat2):
    """
    计算两个经纬度坐标之间的距离(公里)
    使用Haversine公式计算球面距离
    """
    # 将经纬度转换为弧度
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # 地球平均半径(公里)
    return c * r


def build_neighborhood_graph_with_dist(station_data, n_neighbours):
    """
    构建带距离的8邻域图
    :param station_data: 监测站数据字典 {站名: (经度, 纬度)}
    :return: 邻接表表示的图，包含距离信息
    """
    graph = defaultdict(dict)
    station_names = list(station_data.keys())

    for i, name1 in enumerate(station_names):
        lon1, lat1 = station_data[name1]
        distances = []

        # 计算到所有其他站点的距离
        for j, name2 in enumerate(station_names):
            if i == j:
                continue
            lon2, lat2 = station_data[name2]
            distance = haversine(lon1, lat1, lon2, lat2)
            distances.append((distance, name2))

        # 按距离排序并选择最近的8个站点
        distances.sort()
        for dist, name2 in distances[:n_neighbours]:
            graph[name1][name2] = dist

    return graph


# 构建8邻域图
air_quality_graph = build_neighborhood_graph_with_dist(stations, n_neighbours=4)

# 打印结果示例
print("北京市空气质量监测站8邻域图(带距离，单位:公里):")
for station, neighbors in air_quality_graph.items():
    print(f"\n监测站 {station} (位置: {stations[station]}) 的8个最近邻站:")
    for neighbor, dist in neighbors.items():
        print(f"  -> {neighbor}: 距离 {dist:.2f} 公里")

# 可以保存到文件
import json

with open("neighbor_graph.json", "w") as f:
    json.dump(air_quality_graph, f, indent=2, ensure_ascii=False)
