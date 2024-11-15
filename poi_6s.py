import numpy as np
import pandas as pd
import os
from Py6S import *
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# 设定路径
h8_dailybest_path = 'data/h8dailybest/'
h8_data_path = 'D:/H8_data/h8l1/'
mod08_path = 'data/poi_mod08/'
lucc_path = 'data/mcd12q1/'
output_path = 'output/poi_6s/'
station_coords_file = 'Air_stations_lon_lat.csv'

# 波段字典
band_wavelengths = [0.47, 0.51, 0.64, 0.86, 1.6, 2.3]
lucc_to_aerosol = {
    1: AeroProfile.Continental,
    2: AeroProfile.Continental,
    3: AeroProfile.Continental,
    13: AeroProfile.Urban,
    17: AeroProfile.Maritime,
    255: AeroProfile.Maritime
}

# 读取站点经纬度，去除重复的站点
stations = pd.read_csv(station_coords_file, header=None, names=['Station', 'Lon', 'Lat']).drop_duplicates(subset='Station')
station_dict = stations.set_index('Station').T.to_dict()

# 检查输出目录是否存在，不存在则创建
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 处理日期范围
start_date = datetime(2015, 7, 11)
end_date = datetime(2021, 12, 31)

def get_current_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 主函数，遍历每一天的数据
def process_poi_for_date(date):
    date_str = date.strftime("%Y%m%d")  # 格式化日期

    # 读取Himawari-8 dailybest文件
    h8_dailybest_file = os.path.join(h8_dailybest_path, f'poi_h8dailybest_{date_str}.csv')
    if not os.path.exists(h8_dailybest_file):
        print(f"[{get_current_timestamp()}] Date {date_str}: Missing Himawari-8 dailybest file. Assigning -1 for all stations.")
        return

    try:
        h8_dailybest = pd.read_csv(h8_dailybest_file)
    except Exception as e:
        print(f"[{get_current_timestamp()}] Date {date_str}: Error reading Himawari-8 dailybest file: {e}")
        return

    # 读取MOD08数据文件
    mod08_file = os.path.join(mod08_path, f'poi_mod08_{date_str}.csv')
    if not os.path.exists(mod08_file) or (datetime(2016, 2, 19) <= date <= datetime(2016, 2, 27)):
        mod08_data = None
    else:
        try:
            mod08_data = pd.read_csv(mod08_file)
        except Exception as e:
            print(f"[{get_current_timestamp()}] Date {date_str}: Error reading MOD08 file: {e}")
            mod08_data = None

    # 读取LUCC文件，按年份选择
    lucc_file = os.path.join(lucc_path, f'lucc_{date.year}.csv')
    if not os.path.exists(lucc_file):
        lucc_data = None
    else:
        try:
            lucc_data = pd.read_csv(lucc_file)
        except Exception as e:
            print(f"[{get_current_timestamp()}] Date {date_str}: Error reading LUCC file: {e}")
            lucc_data = None

    results = []
    station_count = 0  # 记录处理的站点数
    total_stations = len(h8_dailybest)

    # 开始处理站点
    for _, row in h8_dailybest.iterrows():
        station = row['Station']
        best_time = row['BestTime']
        soz = row['SOZ']

        if best_time == -1 or soz == -1:
            # 无效站点
            results.append([station] + [-1] * 6)
            station_count += 1
            if station_count % 50 == 0:
                print(f"[{get_current_timestamp()}] Date {date_str}: Processed {station_count}/{total_stations} stations (recent 50 stations in 0 seconds).")
            continue

        # 确保 best_time 是 4 位数的字符串
        try:
            best_time_str = str(int(best_time)).zfill(4)
        except ValueError:
            results.append([station] + [-1] * 6)
            station_count += 1
            if station_count % 50 == 0:
                print(f"[{get_current_timestamp()}] Date {date_str}: Processed {station_count}/{total_stations} stations (recent 50 stations in 0 seconds).")
            continue

        # 生成对应时刻的Himawari-8数据文件路径
        h8_file = os.path.join(h8_data_path, date_str, f'poi_h8l1_{date_str}_{best_time_str}.csv')
        if not os.path.exists(h8_file):
            results.append([station] + [-1] * 6)
            station_count += 1
            if station_count % 50 == 0:
                print(f"[{get_current_timestamp()}] Date {date_str}: Processed {station_count}/{total_stations} stations (recent 50 stations in 0 seconds).")
            continue

        try:
            h8_data = pd.read_csv(h8_file)
            h8_row = h8_data[h8_data['Station'] == station].iloc[0]
        except Exception:
            results.append([station] + [-1] * 6)
            station_count += 1
            if station_count % 50 == 0:
                print(f"[{get_current_timestamp()}] Date {date_str}: Processed {station_count}/{total_stations} stations (recent 50 stations in 0 seconds).")
            continue

        # 读取MOD08数据
        if mod08_data is not None:
            try:
                mod08_row = mod08_data[mod08_data['Station'] == station].iloc[0]
                water_vapor = mod08_row['Water_Vapor']
                ozone = mod08_row['Ozone']
            except IndexError:
                water_vapor, ozone = -1, -1
        else:
            water_vapor, ozone = -1, -1

        # 读取LUCC数据
        if lucc_data is not None:
            try:
                lucc_value = lucc_data[lucc_data['Station'] == station]['LUCC_value'].values[0]
                aerosol_profile = lucc_to_aerosol.get(lucc_value, AeroProfile.Continental)
            except IndexError:
                aerosol_profile = AeroProfile.Continental
        else:
            aerosol_profile = AeroProfile.Continental

        # Py6S模型进行大气校正
        try:
            s = SixS()
            s.geometry.solar_z = h8_row['SOZ']
            s.geometry.solar_a = h8_row['SAA']
            s.geometry.view_z = h8_row['SOA']
            s.geometry.view_a = h8_row['SAZ']
            s.aero_profile = aerosol_profile
            s.atmos_profile = AtmosProfile.UserWaterAndOzone(water=water_vapor, ozone=ozone)

            corrected_albedos = []
            start_time = time.time()
            for i, wavelength in enumerate(band_wavelengths):
                s.wavelength = Wavelength(wavelength)
                s.ground_reflectance = GroundReflectance.HomogeneousLambertian(h8_row[f'Albedo_0{i+1}'])
                s.run()
                corrected_albedos.append(s.outputs.pixel_reflectance)
            end_time = time.time()

            results.append([station] + corrected_albedos)
        except Exception:
            results.append([station] + [-1] * 6)

        station_count += 1
        if station_count % 50 == 0:
            duration = int(end_time - start_time) if 'end_time' in locals() else 0
            print(f"[{get_current_timestamp()}] Date {date_str}: Processed {station_count}/{total_stations} stations (recent 50 stations in {duration} seconds).")

    # 保存结果
    if results:
        try:
            results_df = pd.DataFrame(results, columns=['Station'] + [f'Albedo_0{i+1}' for i in range(6)])
            results_file = os.path.join(output_path, f'poi_6s_{date_str}.csv')
            results_df.to_csv(results_file, index=False)
        except Exception as e:
            print(f"[{get_current_timestamp()}] Date {date_str}: Error saving results: {e}")

    print(f"[{get_current_timestamp()}] Date {date_str}: Finished processing. Results saved to {results_file if results else 'N/A'}.")

# 多进程处理主函数
def process_all_dates(start_date, end_date):
    # 生成所有需要处理的日期列表
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date)
        current_date += timedelta(days=1)

    total_dates = len(date_list)
    print(f"Starting processing of {total_dates} dates from {start_date.strftime('%Y%m%d')} to {end_date.strftime('%Y%m%d')}.")

    # 使用 ProcessPoolExecutor 开启4个进程并行处理
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(process_poi_for_date, date): date for date in date_list}
        for future in as_completed(futures):
            date = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[{get_current_timestamp()}] Error processing date {date.strftime('%Y%mdd')}: {e}")

    print("All dates have been processed.")

if __name__ == "__main__":
    process_all_dates(start_date, end_date)
