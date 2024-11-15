import os
import pandas as pd
import numpy as np
from tqdm import tqdm  # 用于显示进度条

# 定义文件路径
L1_PATH = "D:\\H8_data\\h8l1\\"
L2_PATH = "D:\\H8_data\\h8l2arp\\"
OUTPUT_PATH = "D:\\H8_data\\h8dailybest\\"

# 日期范围
date_range = pd.date_range("20150707", "20211231", freq='D')


# 函数：生成有效的时间序列（每10分钟从0100到0850，跳过0240）
def valid_time_sequence():
    times = []
    for hour in range(1, 9):  # 从 0100 到 0850
        for minute in range(0, 60, 10):  # 每10分钟一个时间点
            if hour == 2 and minute == 40:  # 跳过0240时刻
                continue
            times.append(f"{hour:02d}{minute:02d}")
    return times


def load_l2_data(yyyymmdd, hhmm_str):
    l2_file = os.path.join(L2_PATH, yyyymmdd, f"poi_h8l2arp_{yyyymmdd}_{hhmm_str}.csv")

    if not os.path.exists(l2_file):
        print(f"L2 data not found for {yyyymmdd} at {hhmm_str}, skipping this time.")
        return None

    l2_data = pd.read_csv(l2_file)
    l2_data['usable'] = (l2_data['Data Availability'] == 0) & (l2_data['Land/Water Flag'] == 0) & (
                l2_data['Cloud Flag'] == 0)
    return l2_data


def load_l1_data(yyyymmdd, hhmm_str):
    l1_file = os.path.join(L1_PATH, yyyymmdd, f"poi_h8l1_{yyyymmdd}_{hhmm_str}.csv")

    if not os.path.exists(l1_file):
        print(f"L1 data not found for {yyyymmdd} at {hhmm_str}, skipping this time.")
        return None

    l1_data = pd.read_csv(l1_file)
    return l1_data


def process_day(date):
    yyyymmdd = date.strftime('%Y%m%d')

    daily_best = []

    # 有效的时间序列
    valid_times = valid_time_sequence()

    # 遍历每个站点
    stations_checked = False  # 用于检查站点是否存在
    for hhmm_str in valid_times:  # 遍历所有有效时刻
        # 加载L2数据
        l2_data = load_l2_data(yyyymmdd, hhmm_str)
        if l2_data is None:
            continue

        # 加载L1数据
        l1_data = load_l1_data(yyyymmdd, hhmm_str)
        if l1_data is None:
            continue  # 如果没有L1数据，跳过该时刻

        if not stations_checked:
            # 仅在第一个时刻，初始化每日结果（用于保存结果的表）
            stations = l1_data['Station'].unique()
            daily_best = pd.DataFrame({
                'Station': stations,
                'BestTime': np.nan,
                'SOZ': np.inf  # 初始值为无穷大，用于后续比较SOZ
            })
            daily_best['BestTime'] = daily_best['BestTime'].astype(str)  # 将 BestTime 列转换为字符串类型
            stations_checked = True

        # 合并L1和L2数据
        merged_data = pd.merge(l1_data, l2_data[['Station', 'usable']], on='Station')
        usable_data = merged_data[merged_data['usable']]  # 仅保留L2可用的站点

        # 遍历可用的站点，找到SOZ最小的时刻
        for index, row in usable_data.iterrows():
            station = row['Station']
            current_soz = row['SOZ']

            # 找到该站点最小的SOZ并更新
            if current_soz < daily_best.loc[daily_best['Station'] == station, 'SOZ'].values[0]:
                daily_best.loc[daily_best['Station'] == station, 'SOZ'] = current_soz
                daily_best.loc[daily_best['Station'] == station, 'BestTime'] = hhmm_str

    # 将没有可用数据（SOZ为无穷大）的站点替换为-1
    daily_best['SOZ'].replace(np.inf, -1, inplace=True)
    daily_best['BestTime'].replace('nan', '-1', inplace=True)

    # 保存每日最佳结果
    output_file = os.path.join(OUTPUT_PATH, f"poi_h8dailybest_{yyyymmdd}.csv")
    daily_best.to_csv(output_file, index=False)
    print(f"Processed {yyyymmdd} and saved result to {output_file}")


def main():
    for date in tqdm(date_range, desc="Processing days"):
        process_day(date)


if __name__ == "__main__":
    main()
