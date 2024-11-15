import ee
import pandas as pd
import datetime
import os
import math
import logging
from tqdm import tqdm
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def GEE_authorizing():
    service_account = "lobstyu@premium-cipher-424203-d0.iam.gserviceaccount.com"
    credentials_path = 'premium-cipher-424203-d0-c6894a29d00c.json'
    try:
        credentials = ee.ServiceAccountCredentials(service_account, credentials_path)
        ee.Initialize(credentials)
        logging.info("GEE 初始化成功")
    except Exception as e:
        logging.error(f"GEE 初始化失败: {e}")
        raise

def read_poi(csv_path):
    """
    读取 POI 列表，假设 CSV 文件没有标题行，格式为: Station, lon, lat
    """
    try:
        df = pd.read_csv(csv_path, header=None, names=['Station', 'lon', 'lat'])
        logging.info(f"成功读取 POI 数据，共 {len(df)} 个 POI")
        return df
    except Exception as e:
        logging.error(f"读取 POI 文件失败: {e}")
        raise

def create_feature_collection(poi_df):
    """
    将 POI DataFrame 转换为 GEE FeatureCollection
    """
    try:
        features = []
        for _, row in poi_df.iterrows():
            feature = ee.Feature(ee.Geometry.Point([row['lon'], row['lat']]), {
                'Station': row['Station']
            })
            features.append(feature)
        fc = ee.FeatureCollection(features)
        logging.info("成功创建 FeatureCollection")
        return fc
    except Exception as e:
        logging.error(f"创建 FeatureCollection 失败: {e}")
        raise

def get_date_list(start_date, end_date):
    """
    生成日期列表，格式为 'YYYY-MM-DD'
    """
    try:
        start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        delta = end - start
        date_list = [(start + datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(delta.days + 1)]
        logging.info(f"生成日期列表，从 {start_date} 到 {end_date}，共 {len(date_list)} 天")
        return date_list
    except Exception as e:
        logging.error(f"生成日期列表失败: {e}")
        raise

def ensure_directory(path):
    """
    确保目录存在，不存在则创建
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
            logging.info(f"创建目录: {path}")
        except Exception as e:
            logging.error(f"创建目录失败: {e}")
            raise

def process_date(date, collection, bands, poi_fc, output_dir, max_retries=3, backoff_factor=0.5):
    """
    处理单个日期的数据获取和保存，包含重试机制
    """
    for attempt in range(1, max_retries + 1):
        try:
            logging.info(f"开始处理日期: {date}")

            # 过滤日期
            next_day = (datetime.datetime.strptime(date, '%Y-%m-%d') + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            image = collection.filterDate(date, next_day).first()

            if image is None:
                logging.warning(f"{date} 没有数据")
                return f"{date}: No data"

            # 选择所需的波段
            image = image.select(bands)

            # 计算总风速: sqrt(u^2 + v^2)
            wind_u = image.select('u_component_of_wind_10m')
            wind_v = image.select('v_component_of_wind_10m')
            wind_speed = wind_u.pow(2).add(wind_v.pow(2)).sqrt().rename('wind_speed')

            # 添加 wind_speed 到影像
            image = image.addBands(wind_speed)

            # 选择最终需要的气象要素
            final_bands = [
                'dewpoint_temperature_2m',
                'temperature_2m',
                'surface_pressure',
                'total_precipitation_sum',
                'wind_speed'
            ]
            image = image.select(final_bands)

            # 采样 POI
            sampled = image.sampleRegions(
                collection=poi_fc,
                scale=1000,  # ERA5-Land 的空间分辨率约为 0.1 度 (~10km)
                geometries=False
            )

            # 获取结果
            sampled_dict = sampled.getInfo()

            # 检查是否有采样结果
            if 'features' not in sampled_dict or len(sampled_dict['features']) == 0:
                logging.warning(f"{date} 没有采样结果")
                return f"{date}: No samples"

            # 转换为 DataFrame
            records = []
            for feature in sampled_dict['features']:
                props = feature['properties']
                record = {
                    'Station': props.get('Station', ''),
                    'dewpoint_temperature_2m': props.get('dewpoint_temperature_2m', None),
                    'temperature_2m': props.get('temperature_2m', None),
                    'surface_pressure': props.get('surface_pressure', None),
                    'total_precipitation_sum': props.get('total_precipitation_sum', None),
                    'wind_10m_max': props.get('wind_speed', None)
                }
                records.append(record)

            df_day = pd.DataFrame(records)

            # 添加日期列
            df_day['date'] = date.replace('-', '')

            # 定义文件名
            filename = f"poi_era5_{date.replace('-', '')}.csv"
            filepath = os.path.join(output_dir, filename)

            # 保存为 CSV
            df_day.to_csv(filepath, index=False)

            logging.info(f"完成处理日期: {date}")
            return f"{date}: Success"

        except Exception as e:
            logging.error(f"处理日期 {date} 时出错: {e}")
            if attempt < max_retries:
                sleep_time = backoff_factor * (2 ** (attempt - 1))
                logging.info(f"重试处理日期 {date}，第 {attempt} 次重试，等待 {sleep_time} 秒")
                time.sleep(sleep_time)
            else:
                logging.error(f"处理日期 {date} 失败，达到最大重试次数")
                return f"{date}: Failed after {max_retries} attempts"

def main():
    # 初始化 GEE
    GEE_authorizing()

    # 读取 POI 列表
    poi_csv_path = 'Air_stations_lon_lat.csv'
    poi_df = read_poi(poi_csv_path)

    # 创建 FeatureCollection
    poi_fc = create_feature_collection(poi_df)

    # 定义日期范围
    start_date = '2015-07-07'
    end_date = '2021-12-31'
    date_list = get_date_list(start_date, end_date)

    # 定义输出目录
    output_dir = os.path.join('downloaded_data', 'era5')
    ensure_directory(output_dir)

    # 定义 GEE ImageCollection 和波段
    collection = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
    bands = [
        'dewpoint_temperature_2m',
        'temperature_2m',
        'surface_pressure',
        'total_precipitation_sum',
        'u_component_of_wind_10m',
        'v_component_of_wind_10m'
    ]

    # 使用 tqdm 显示进度条
    with tqdm(total=len(date_list), desc="处理进度") as pbar:
        for date in date_list:
            result = process_date(date, collection, bands, poi_fc, output_dir)
            logging.info(result)
            pbar.update(1)

if __name__ == "__main__":
    main()
