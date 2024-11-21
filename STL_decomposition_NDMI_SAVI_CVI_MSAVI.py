# STL_decomposition_new_indices.py

import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.signal import savgol_filter
from statsmodels.tsa.seasonal import STL
from datetime import datetime

# 定义波段波长
band_wavelengths = [0.47, 0.51, 0.64, 0.86, 1.6, 2.3]


def compute_ndmi(albedo_nir, albedo_swir):
    """
    计算NDMI值，并生成掩膜标记无效值。
    无效值和缺失值在掩膜中标记为True。

    返回:
        ndmi (np.ndarray): 计算得到的NDMI值，非法值为NaN
        mask (np.ndarray): 掩膜，非法值为True
    """
    invalid = (albedo_nir == -1) | (albedo_swir == -1)
    ndmi = (albedo_nir - albedo_swir) / (albedo_nir + albedo_swir)
    ndmi[invalid] = np.nan
    return ndmi, invalid


def compute_savi(albedo_red, albedo_nir, L=0.5):
    """
    计算SAVI值，并生成掩膜标记无效值。
    无效值和缺失值在掩膜中标记为True。

    参数:
        albedo_red (np.ndarray): Red波段Albedo数据
        albedo_nir (np.ndarray): NIR波段Albedo数据
        L (float): 土壤亮度校正因子

    返回:
        savi (np.ndarray): 计算得到的SAVI值，非法值为NaN
        mask (np.ndarray): 掩膜，非法值为True
    """
    invalid = (albedo_red == -1) | (albedo_nir == -1)
    savi = ((albedo_nir - albedo_red) / (albedo_nir + albedo_red + L)) * (1 + L)
    savi[invalid] = np.nan
    return savi, invalid


def compute_cvi(albedo_red, albedo_nir):
    """
    计算CVI值，并生成掩膜标记无效值。
    无效值和缺失值在掩膜中标记为True。

    参数:
        albedo_red (np.ndarray): Red波段Albedo数据
        albedo_nir (np.ndarray): NIR波段Albedo数据

    返回:
        cvi (np.ndarray): 计算得到的CVI值，非法值为NaN
        mask (np.ndarray): 掩膜，非法值为True
    """
    invalid = (albedo_red == -1) | (albedo_nir == -1)
    with np.errstate(divide='ignore', invalid='ignore'):
        cvi = (albedo_nir / albedo_red) - 1
    cvi[invalid] = np.nan
    return cvi, invalid


def compute_msavi(albedo_red, albedo_nir):
    """
    计算MSAVI值，并生成掩膜标记无效值。
    无效值和缺失值在掩膜中标记为True。

    参数:
        albedo_red (np.ndarray): Red波段Albedo数据
        albedo_nir (np.ndarray): NIR波段Albedo数据

    返回:
        msavi (np.ndarray): 计算得到的MSAVI值，非法值为NaN
        mask (np.ndarray): 掩膜，非法值为True
    """
    invalid = (albedo_red == -1) | (albedo_nir == -1)
    term = (2 * albedo_nir + 1)**2 - 8 * (albedo_nir - albedo_red)
    with np.errstate(invalid='ignore'):
        sqrt_term = np.sqrt(term)
    msavi = (2 * albedo_nir + 1 - sqrt_term) / 2
    msavi[invalid] = np.nan
    return msavi, invalid


def get_sorted_csv_files(input_dir):
    """
    获取按日期排序的所有CSV文件路径

    参数:
        input_dir (str): 输入目录路径

    返回:
        csv_files_sorted (list): 按日期排序的CSV文件路径列表
        dates (list): 对应的日期列表，格式为YYYYMMDD
    """
    csv_files = glob.glob(os.path.join(input_dir, 'poi_6s_*.csv'))

    # 提取日期并排序
    def extract_date(file_path):
        basename = os.path.basename(file_path)
        date_str = basename.replace('poi_6s_', '').replace('.csv', '')
        return datetime.strptime(date_str, '%Y%m%d')

    csv_files_sorted = sorted(csv_files, key=extract_date)
    dates = [extract_date(f).strftime('%Y%m%d') for f in csv_files_sorted]
    return csv_files_sorted, dates


def build_indices_dataframe(csv_files_sorted, dates):
    """
    遍历所有CSV文件，计算NDMI、SAVI、CVI和MSAVI，并构建相应的DataFrame，同时构建掩膜DataFrame

    参数:
        csv_files_sorted (list): 排序后的CSV文件路径列表
        dates (list): 对应的日期列表

    返回:
        indices_df (dict): 包含各指数DataFrame的字典
            keys: 'NDMI', 'SAVI', 'CVI', 'MSAVI'
            values: pd.DataFrame 对应指数的值
        mask_df (dict): 包含各指数掩膜DataFrame的字典
            keys: 'NDMI', 'SAVI', 'CVI', 'MSAVI'
            values: pd.DataFrame 对应指数的掩膜
        stations (list): 站点列表
    """
    # 读取第一个文件以获取站点列表
    first_df = pd.read_csv(csv_files_sorted[0])
    stations = first_df['Station'].tolist()

    # 初始化DataFrame
    indices = ['NDMI', 'SAVI', 'CVI', 'MSAVI']
    indices_df = {index: pd.DataFrame(index=dates, columns=stations, dtype=float) for index in indices}
    mask_df = {index: pd.DataFrame(index=dates, columns=stations, dtype=bool) for index in indices}

    # 遍历所有CSV文件并计算各指数
    for file, date in tqdm(zip(csv_files_sorted, dates), total=len(csv_files_sorted), desc='Processing CSV files'):
        df = pd.read_csv(file)

        # 检查实际列名
        expected_columns = [f'Albedo_{i:02d}' for i in range(1, 7)]
        actual_columns = df.columns.tolist()
        missing_columns = [col for col in expected_columns if col not in actual_columns]
        if missing_columns:
            raise KeyError(f"缺少预期的列: {missing_columns} 在文件 {file}")

        # 根据波段波长映射 Albedo_* 列
        # 假设：
        # Albedo_01: 0.47 (Blue)
        # Albedo_02: 0.51 (Green)
        # Albedo_03: 0.64 (Red)
        # Albedo_04: 0.86 (NIR)
        # Albedo_05: 1.6 (SWIR1)
        # Albedo_06: 2.3 (SWIR2)

        albedo_blue = df['Albedo_01'].values
        albedo_green = df['Albedo_02'].values
        albedo_red = df['Albedo_03'].values
        albedo_nir = df['Albedo_04'].values
        albedo_swir = df['Albedo_05'].values

        # 计算各指数
        ndmi, ndmi_mask = compute_ndmi(albedo_nir, albedo_swir)
        savi, savi_mask = compute_savi(albedo_red, albedo_nir)
        cvi, cvi_mask = compute_cvi(albedo_red, albedo_nir)
        msavi, msavi_mask = compute_msavi(albedo_red, albedo_nir)

        # 将结果存入DataFrame
        indices_df['NDMI'].loc[date] = ndmi
        mask_df['NDMI'].loc[date] = ndmi_mask

        indices_df['SAVI'].loc[date] = savi
        mask_df['SAVI'].loc[date] = savi_mask

        indices_df['CVI'].loc[date] = cvi
        mask_df['CVI'].loc[date] = cvi_mask

        indices_df['MSAVI'].loc[date] = msavi
        mask_df['MSAVI'].loc[date] = msavi_mask

    # 转换索引为datetime类型
    for index in indices:
        indices_df[index].index = pd.to_datetime(indices_df[index].index, format='%Y%m%d')
        mask_df[index].index = pd.to_datetime(mask_df[index].index, format='%Y%m%d')

    return indices_df, mask_df, stations


def interpolate_series(series):
    """
    对时间序列进行插值，填补缺失值

    参数:
        series (pd.Series): 时间序列

    返回:
        interpolated (pd.Series): 插值后的时间序列
    """
    interpolated = series.interpolate(method='time', limit_direction='both')
    return interpolated


def smooth_series(series, window_length=51, polyorder=3):
    """
    使用Savitzky-Golay滤波器对数据进行平滑

    参数:
        series (pd.Series): 插值后的时间序列
        window_length (int): 滑动窗口长度
        polyorder (int): 多项式阶数

    返回:
        smoothed (np.ndarray): 平滑后的数据
    """
    # 确保window_length为奇数且不超过数据长度
    if len(series) < window_length:
        window_length = len(series) if len(series) % 2 != 0 else len(series) - 1
    if window_length < 5:
        # 如果数据点过少，返回原始数据
        return series.values
    smoothed = savgol_filter(series.values, window_length=window_length, polyorder=polyorder, mode='interp')
    return smoothed


def perform_stl_decomposition(series, period=365):
    """
    对平滑后的时间序列进行STL分解

    参数:
        series (pd.Series): 平滑后的时间序列
        period (int): 季节性周期

    返回:
        trend (np.ndarray): 趋势分量
        seasonal (np.ndarray): 季节性分量
        resid (np.ndarray): 残差分量
    """
    stl = STL(series, period=period, robust=True)
    result = stl.fit()
    return result.trend, result.seasonal, result.resid


def apply_mask_to_residual(residual, mask_series):
    """
    在残差中应用掩膜，将无效值位置赋值为-1

    参数:
        residual (np.ndarray): 残差分量
        mask_series (pd.Series): 掩膜系列，True表示无效或缺失

    返回:
        masked_residual (np.ndarray): 应用掩膜后的残差
    """
    masked_residual = residual.copy()
    # 将掩膜为True的位置赋值为-1
    masked_residual[mask_series.values] = -1
    return masked_residual


def save_decomposition(station, dates, index_name, original, trend, seasonal, residual, output_dir):
    """
    将STL分解结果保存为CSV文件

    参数:
        station (str): 站点名称
        dates (pd.DatetimeIndex): 日期索引
        index_name (str): 指数名称（NDMI, SAVI, CVI, MSAVI）
        original (np.ndarray): 原始指数值
        trend (np.ndarray): 趋势分量
        seasonal (np.ndarray): 季节性分量
        residual (np.ndarray): 残差分量
        output_dir (str): 输出目录
    """
    output_df = pd.DataFrame({
        'Date': dates.strftime('%Y%m%d'),
        index_name: original,
        'trend': trend,
        'seasonal': seasonal,
        'residual': residual
    })
    output_file = os.path.join(output_dir, f"{station}_STL_{index_name}_20150707_20211231.csv")
    output_df.to_csv(output_file, index=False)


def process_index(station, series, mask_series, index_name, output_dir):
    """
    处理单个指数的时间序列，包括插值、平滑、STL分解、掩膜应用和保存结果

    参数:
        station (str): 站点名称
        series (pd.Series): 指数时间序列
        mask_series (pd.Series): 掩膜系列
        index_name (str): 指数名称（NDMI, SAVI, CVI, MSAVI）
        output_dir (str): 输出目录
    """
    # 插值处理
    interpolated = interpolate_series(series)

    if interpolated.isna().all():
        print(f"所有{index_name}值缺失，跳过站点 {station}")
        return

    # 平滑处理
    smoothed = smooth_series(interpolated)
    smoothed_series = pd.Series(smoothed, index=interpolated.index)

    # STL分解
    try:
        trend, seasonal, residual = perform_stl_decomposition(smoothed_series)
    except Exception as e:
        print(f"STL分解失败，站点 {station}，指数 {index_name}，错误: {e}")
        return

    # 应用掩膜到残差
    masked_residual = apply_mask_to_residual(residual, mask_series)

    # 保存结果
    save_decomposition(
        station,
        smoothed_series.index,
        index_name,
        smoothed_series.values,
        trend,
        seasonal,
        masked_residual,
        output_dir
    )


def main():
    # 定义输入和输出目录
    input_dir = r'D:\H8_data\poi_6s'
    output_dirs = {
        'NDMI': 'STL_decomposition_NDMI',
        'SAVI': 'STL_decomposition_SAVI',
        'CVI': 'STL_decomposition_CVI',
        'MSAVI': 'STL_decomposition_MSAVI'
    }
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # 获取所有CSV文件并排序
    csv_files_sorted, dates = get_sorted_csv_files(input_dir)

    # 构建各指数的DataFrame和掩膜DataFrame
    indices_df, mask_df, stations = build_indices_dataframe(csv_files_sorted, dates)

    # 遍历每个站点进行处理
    for station in tqdm(stations, desc='Processing stations'):
        for index_name in ['NDMI', 'SAVI', 'CVI', 'MSAVI']:
            station_series = indices_df[index_name][station]
            station_mask = mask_df[index_name][station]

            process_index(
                station,
                station_series,
                station_mask,
                index_name,
                output_dirs[index_name]
            )


if __name__ == '__main__':
    main()
