import pandas as pd
import numpy as np
import netCDF4 as nc
from osgeo import gdal

# 加载站点信息
stations_df = pd.read_csv('Air_stations_lon_lat.csv', header=None, names=['Station', 'Lon', 'Lat'])

# 读取 NetCDF 文件的经纬度
def load_nc_lonlat(file_path):
    dataset = nc.Dataset(file_path, 'r')
    lon = dataset.variables['longitude'][:]
    lat = dataset.variables['latitude'][:]
    dataset.close()
    return lon, lat

# 从 NetCDF 文件中计算站点的像素坐标
def lonlat_to_pixel(lon, lat, station_lon, station_lat):
    lon_idx = np.abs(lon - station_lon).argmin()
    lat_idx = np.abs(lat - station_lat).argmin()
    return lon_idx, lat_idx

# 加载 H8L1 和 L2ARP 的经纬度信息
h8l1_lon, h8l1_lat = load_nc_lonlat(
    'D:\\Users\\laobi\\PycharmProjects\\H8_POI_preprocessing\\downloaded_data\\h8l1\\himawari_20150715_04.nc')
l2arp_lon, l2arp_lat = load_nc_lonlat(
    'D:\\Users\\laobi\\PycharmProjects\\H8_POI_preprocessing\\downloaded_data\\h8l2arp\\himawari_20150715_04.nc')

# 读取 MOD08 的 XDim 和 YDim 信息
def load_mod08_lonlat(mod08_file):
    dataset = nc.Dataset(mod08_file, 'r')
    mod08_lon = dataset.variables['XDim'][:]
    mod08_lat = dataset.variables['YDim'][:]
    dataset.close()
    return mod08_lon, mod08_lat

mod08_lon, mod08_lat = load_mod08_lonlat(
    'D:\\Users\\laobi\\PycharmProjects\\H8_POI_preprocessing\\downloaded_data\\mod08\\MOD08_20150715.hdf'
)

# 正确转换站点经纬度到 MOD08 像素坐标
def convert_lonlat_to_mod08_pixel(mod08_lon, mod08_lat, station_lon, station_lat):
    # MODIS 全球数据通常是等经纬度网格的
    # 通过寻找最近的经度和纬度网格点
    lon_idx = np.abs(mod08_lon - station_lon).argmin()
    lat_idx = np.abs(mod08_lat - station_lat).argmin()
    return lon_idx, lat_idx

# 读取 LUCC.tif 的地理变换信息
def load_geotransform(file_path):
    dataset = gdal.Open(file_path)
    if dataset is None:
        raise FileNotFoundError(f"Unable to open {file_path}")
    geotransform = dataset.GetGeoTransform()  # 获取地理变换
    dataset = None  # 关闭文件
    return geotransform

# 计算站点在 LUCC.tif 中的像素坐标
def lonlat_to_pixel_lucc(lon, lat, geotransform):
    origin_x = geotransform[0]
    pixel_width = geotransform[1]
    origin_y = geotransform[3]
    pixel_height = geotransform[5]

    x = int((lon - origin_x) / pixel_width)
    y = int((lat - origin_y) / pixel_height)

    return x, y

# 获取 LUCC.tif 的地理变换信息
lucc_geotransform = load_geotransform(
    'D:\\Users\\laobi\\PycharmProjects\\H8_POI_preprocessing\\preprocessing_data\\2015\\H8_LUCC.tif')

# 创建结果 DataFrame 的行列表
rows = []

# 计算每个站点的像素坐标
for _, row in stations_df.iterrows():
    station = row['Station']
    lon = row['Lon']
    lat = row['Lat']

    # 计算 H8L1 和 L2ARP 的 xy 坐标
    h8l1_x, h8l1_y = lonlat_to_pixel(h8l1_lon, h8l1_lat, lon, lat)
    l2arp_x, l2arp_y = lonlat_to_pixel(l2arp_lon, l2arp_lat, lon, lat)

    # 计算 MOD08 的 XDim 和 YDim 坐标，正确使用经纬度
    mod08_x, mod08_y = convert_lonlat_to_mod08_pixel(mod08_lon, mod08_lat, lon, lat)

    # 计算 LUCC.tif 的 xy 坐标
    lucc_x, lucc_y = lonlat_to_pixel_lucc(lon, lat, lucc_geotransform)

    # 将这一行的数据存储在列表中
    rows.append({
        'Station': station,
        'Lon': lon,
        'Lat': lat,
        'H8L1_x': h8l1_x, 'H8L1_y': h8l1_y,
        'L2ARP_x': l2arp_x, 'L2ARP_y': l2arp_y,
        'MOD08_XDim': mod08_x,
        'MOD08_YDim': mod08_y,
        'LUCC_x': lucc_x,  # LUCC.tif 对应的 x 坐标
        'LUCC_y': lucc_y  # LUCC.tif 对应的 y 坐标
    })

# 使用 pd.DataFrame 将所有行数据组合成一个 DataFrame
results_df = pd.DataFrame(rows)

# 保存结果
results_df.to_csv('station_xy_lookup_table.csv', index=False)
