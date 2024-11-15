# STL_decomposition_new.py

import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.signal import savgol_filter
from statsmodels.tsa.seasonal import STL
from datetime import datetime


def compute_ndvi(albedo_03, albedo_04):
    """
    计算NDVI值，并生成掩膜标记无效值。
    无效值和缺失值在掩膜中标记为True。

    参数:
        albedo_03 (np.ndarray): Albedo 03数据
        albedo_04 (np.ndarray): Albedo 04数据

    返回:
        ndvi (np.ndarray): 计算得到的NDVI值，非法值为NaN
        mask (np.ndarray): 掩膜，非法值为True
    """
    invalid = (albedo_03 == -1) | (albedo_04 == -1)
    ndvi = (albedo_04 - albedo_03) / (albedo_04 + albedo_03)
    ndvi[invalid] = np.nan
    return ndvi, invalid


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


def build_ndvi_dataframe(csv_files_sorted, dates):
    """
    遍历所有CSV文件，计算NDVI并构建DataFrame，同时构建掩膜DataFrame

    参数:
        csv_files_sorted (list): 排序后的CSV文件路径列表
        dates (list): 对应的日期列表

    返回:
        ndvi_df (pd.DataFrame): NDVI值的DataFrame
        mask_df (pd.DataFrame): 掩膜的DataFrame，True表示无效或缺失
        stations (list): 站点列表
    """
    # 读取第一个文件以获取站点列表
    first_df = pd.read_csv(csv_files_sorted[0])
    stations = first_df['Station'].tolist()

    # 初始化NDVI和掩膜DataFrame
    ndvi_df = pd.DataFrame(index=dates, columns=stations, dtype=float)
    mask_df = pd.DataFrame(index=dates, columns=stations, dtype=bool)

    # 遍历所有CSV文件并计算NDVI
    for file, date in tqdm(zip(csv_files_sorted, dates), total=len(csv_files_sorted), desc='Processing CSV files'):
        df = pd.read_csv(file)
        ndvi, mask = compute_ndvi(df['Albedo_03'].values, df['Albedo_04'].values)
        ndvi_series = pd.Series(ndvi, index=df['Station'])
        mask_series = pd.Series(mask, index=df['Station'])
        ndvi_df.loc[date] = ndvi_series
        mask_df.loc[date] = mask_series

    # 转换索引为datetime类型
    ndvi_df.index = pd.to_datetime(ndvi_df.index, format='%Y%m%d')
    mask_df.index = pd.to_datetime(mask_df.index, format='%Y%m%d')
    return ndvi_df, mask_df, stations


def interpolate_ndvi(series):
    """
    对NDVI时间序列进行插值，填补缺失值

    参数:
        series (pd.Series): NDVI时间序列

    返回:
        interpolated (pd.Series): 插值后的时间序列
    """
    interpolated = series.interpolate(method='time', limit_direction='both')
    return interpolated


def smooth_ndvi(series, window_length=51, polyorder=3):
    """
    使用Savitzky-Golay滤波器对NDVI数据进行平滑

    参数:
        series (pd.Series): 插值后的NDVI时间序列
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
    对平滑后的NDVI时间序列进行STL分解

    参数:
        series (pd.Series): 平滑后的NDVI时间序列
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


def save_decomposition(station, dates, ndvi, trend, seasonal, residual, output_dir):
    """
    将STL分解结果保存为CSV文件

    参数:
        station (str): 站点名称
        dates (pd.DatetimeIndex): 日期索引
        ndvi (np.ndarray): 原始NDVI值
        trend (np.ndarray): 趋势分量
        seasonal (np.ndarray): 季节性分量
        residual (np.ndarray): 残差分量
        output_dir (str): 输出目录
    """
    output_df = pd.DataFrame({
        'Date': dates.strftime('%Y%m%d'),
        'NDVI': ndvi,
        'trend': trend,
        'seasonal': seasonal,
        'residual': residual
    })
    output_file = os.path.join(output_dir, f"{station}_STL_20150707_20211231.csv")
    output_df.to_csv(output_file, index=False)


def main():
    # 定义输入和输出目录
    input_dir = r'D:\H8_data\poi_6s'
    output_dir = 'STL_decomposition'
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有CSV文件并排序
    csv_files_sorted, dates = get_sorted_csv_files(input_dir)

    # 构建NDVI DataFrame和掩膜DataFrame
    ndvi_df, mask_df, stations = build_ndvi_dataframe(csv_files_sorted, dates)

    # 遍历每个站点进行处理
    for station in tqdm(stations, desc='Processing stations'):
        station_series = ndvi_df[station]
        station_mask = mask_df[station]

        # 插值处理
        interpolated = interpolate_ndvi(station_series)
        if interpolated.isna().all():
            print(f"所有NDVI值缺失，跳过站点 {station}")
            continue

        # 平滑处理
        smoothed = smooth_ndvi(interpolated)
        smoothed_series = pd.Series(smoothed, index=interpolated.index)

        # STL分解
        try:
            trend, seasonal, residual = perform_stl_decomposition(smoothed_series)
        except Exception as e:
            print(f"STL分解失败，站点 {station}，错误: {e}")
            continue

        # 应用掩膜到残差
        # 由于插值和平滑处理改变了原始数据的位置，需重新对齐掩膜
        masked_residual = apply_mask_to_residual(residual, station_mask)

        # 保存结果
        save_decomposition(
            station,
            smoothed_series.index,
            smoothed_series.values,
            trend,
            seasonal,
            masked_residual,
            output_dir
        )


if __name__ == '__main__':
    main()
