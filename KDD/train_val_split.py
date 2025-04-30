import pandas as pd

# 加载数据 (替换为实际数据加载方式)
def select_data_by_date_range(df, year_ranges, month_ranges=None):
    """
    按年月范围选择数据
    参数:
        df: 原始DataFrame
        year_ranges: 年份范围列表，如[(2017,2017), (2018,2019)]表示选择2017年和2018-2019年
        month_ranges: 月份范围列表，如[(1,6), (9,12)]表示选择1-6月和9-12月
    """
    masks = []
    for start_year, end_year in year_ranges:
        year_mask = (df['utc_time'].dt.year >= start_year) & (df['utc_time'].dt.year <= end_year)
        if month_ranges:
            month_mask = pd.Series(False, index=df.index)
            for start_month, end_month in month_ranges:
                month_mask |= (df['utc_time'].dt.month >= start_month) & (df['utc_time'].dt.month <= end_month)
            masks.append(year_mask & month_mask)
        else:
            masks.append(year_mask)

    combined_mask = pd.Series(False, index=df.index)
    for mask in masks:
        combined_mask |= mask

    return df[combined_mask].copy()

df = pd.read_csv("./cleaned_data/cleaned_aq.csv")

df['utc_time'] = pd.to_datetime(df['utc_time'])

train_df = select_data_by_date_range(df, year_ranges=[(2017, 2017)], month_ranges=[(1,2), (10, 12)])
test_df = select_data_by_date_range(df, year_ranges=[(2018, 2018)], month_ranges=[(1,2)])

# 保存为CSV文件
train_df.to_csv("train/train.csv", index=False)
test_df.to_csv("validate/val.csv", index=False)

print(f"训练集已保存 (共 {len(train_df)} 条记录): train.csv")
print(f"验证集已保存 (共 {len(test_df)} 条记录): val.csv")
print("\n训练集示例:")
print(train_df.head())
print("\n验证集示例:")
print(test_df.head())