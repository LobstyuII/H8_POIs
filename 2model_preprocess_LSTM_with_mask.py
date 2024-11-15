# preprocess_data_with_prior.py

import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from datetime import datetime


def preprocess_data_with_prior(data_dir, output_file='preprocessed_data_with_prior.pt',
                               scaler_file='scaler_prior.joblib',
                               max_post_days=28, pre_ndvi_days=3):
    """
    预处理数据，生成用于f(EEFDR)和f(Non_EEFDR)的特征，并根据平均日数据量进行加权。
    """
    pre_ndvi_list = []
    eefdr_factors_list = []
    noneefdr_factors_list = []
    target_list = []
    mask_list = []
    per_sample_features_list = []
    weights_list = []

    # f(EEFDR)使用的因子（13个因子）
    eefdr_columns = [
        'temperature', 'dewpoint_temperature', 'surface_pressure',
        'wind_speed', 'total_precipitation_sum', 'PM10',
        'PM10_PM25', 'current_precipitation', 'lat', 'lon',
        'LUCC', 'day_of_year_sin', 'day_of_year_cos'
    ]

    # f(Non_EEFDR)使用的因子（剔除LUCC, PM10, PM10_PM25, dry_period）
    noneefdr_columns = [
        'temperature', 'dewpoint_temperature', 'surface_pressure',
        'wind_speed', 'total_precipitation_sum', 'current_precipitation',
        'lat', 'lon', 'day_of_year_sin', 'day_of_year_cos'
    ]

    lucc_categories = [
        "Urban and Built-up Lands",
        "Deciduous Broadleaf Forests",
        "Evergreen Needleleaf Forests",
        "Evergreen Broadleaf Forests",
        "Deciduous Needleleaf Forests",
        "Mixed Forests",
        "Woody Savannas",
        "Savannas",
        "Grasslands",
        "Croplands",
        "Cropland",
        "Natural Vegetation Mosaics",
        "Closed Shrublands",
        "Open Shrublands",
        "Permanent Wetlands",
        "Permanent Snow and Ice",
        "Barren",
        "Water Bodies",
        "Unclassified"
    ]
    le = LabelEncoder()
    le.fit(lucc_categories)

    # 计算每一天的数据量，用于加权
    day_counts = np.zeros(max_post_days)
    total_samples = 0

    # 先统计每一天的数据量
    for file in tqdm(os.listdir(data_dir), desc='Counting day data'):
        if file.endswith('_events.csv'):
            file_path = os.path.join(data_dir, file)
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                try:
                    post_data_str = row['post_data'].replace('""', '"').replace("''", "'")
                    post_data = json.loads(post_data_str)
                    if not post_data:
                        continue
                    post_df = pd.DataFrame(post_data)
                    if 'resNDVI' not in post_df.columns:
                        continue
                    post_df = post_df.replace([-99999, -1], np.nan)
                    post_df = post_df.dropna(subset=['resNDVI'])
                    if post_df.empty:
                        continue
                    actual_length = len(post_df.iloc[:max_post_days])
                    day_counts[:actual_length] += 1
                    total_samples += 1
                except:
                    continue

    # 计算权重（每一天的数据量占总样本数的比例）
    day_weights = day_counts / total_samples
    # 为了避免后面的除法出现除以0的情况，对权重为0的天数赋予一个很小的值
    day_weights[day_weights == 0] = 1e-6
    # 计算每一天的加权系数（取倒数，数据量少的天数权重更大）
    day_weights = 1 / day_weights
    # 归一化
    day_weights = day_weights / np.sum(day_weights)

    total_skipped = 0

    for file in tqdm(os.listdir(data_dir), desc='Preprocessing events'):
        if file.endswith('_events.csv'):
            file_path = os.path.join(data_dir, file)
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                try:
                    post_data_str = row['post_data'].replace('""', '"').replace("''", "'")
                    post_data = json.loads(post_data_str)

                    if not post_data:
                        total_skipped += 1
                        continue

                    post_df = pd.DataFrame(post_data)

                    if 'resNDVI' not in post_df.columns:
                        total_skipped += 1
                        continue

                    post_df = post_df.replace([-99999, -1], np.nan)
                    post_df = post_df.dropna(subset=['resNDVI'])

                    if post_df.empty:
                        total_skipped += 1
                        continue

                    post_df = post_df.iloc[:max_post_days]
                    actual_length = len(post_df)

                    if actual_length < max_post_days:
                        pad_length = max_post_days - actual_length
                        pad_data = {
                            'resNDVI': 0.0,
                            'temperature': 0.0,
                            'dewpoint_temperature': 0.0,
                            'surface_pressure': 0.0,
                            'wind_speed': 0.0,
                            'total_precipitation_sum': 0.0,
                            'LUCC': 'Unclassified',
                            'PM10': 0.0,
                            'PM10_PM25': 0.0,
                            'lon': 0.0,
                            'lat': 0.0,
                            'current_precipitation': 0.0,
                            'date': '19700101'
                        }
                        pad_df = pd.DataFrame([pad_data] * pad_length)
                        post_df = pd.concat([post_df, pad_df], ignore_index=True)

                    post_df = post_df.iloc[:max_post_days]
                    post_df['LUCC'] = post_df['LUCC'].fillna('Unclassified')
                    post_df['LUCC_encoded'] = le.transform(post_df['LUCC'])
                    post_df['date'] = pd.to_datetime(post_df['date'], format='%Y%m%d', errors='coerce')
                    post_df['day_of_year'] = post_df['date'].dt.dayofyear.fillna(1).astype(int)
                    post_df['day_of_year_sin'] = np.sin(2 * np.pi * post_df['day_of_year'] / 365.0)
                    post_df['day_of_year_cos'] = np.cos(2 * np.pi * post_df['day_of_year'] / 365.0)

                    post_df = post_df.fillna(0.0)

                    # 生成f(EEFDR)的特征
                    eefdr_features = post_df[[
                        'temperature', 'dewpoint_temperature', 'surface_pressure',
                        'wind_speed', 'total_precipitation_sum', 'PM10',
                        'PM10_PM25', 'current_precipitation', 'lat', 'lon',
                        'LUCC_encoded', 'day_of_year_sin', 'day_of_year_cos'
                    ]].values.astype(np.float32)  # [28,13]

                    # 生成f(Non_EEFDR)的特征
                    noneefdr_features = post_df[[
                        'temperature', 'dewpoint_temperature', 'surface_pressure',
                        'wind_speed', 'total_precipitation_sum', 'current_precipitation',
                        'lat', 'lon', 'day_of_year_sin', 'day_of_year_cos'
                    ]].values.astype(np.float32)  # [28,10]

                    # 创建掩码
                    mask = np.zeros((max_post_days,), dtype=np.float32)
                    mask[:actual_length] = 1.0  # [28,]

                    pre_ndvi = [
                        row['pre_NDVI_residual_day1'],
                        row['pre_NDVI_residual_day2'],
                        row['pre_NDVI_residual_day3']
                    ]
                    pre_ndvi = np.array(pre_ndvi, dtype=np.float32)
                    pre_ndvi = np.nan_to_num(pre_ndvi, nan=0.0, posinf=0.0, neginf=0.0)

                    per_sample_features = [
                        row['current_precipitation'],
                        row['dry_period']
                    ]
                    per_sample_features = np.array(per_sample_features, dtype=np.float32)
                    per_sample_features = np.nan_to_num(per_sample_features, nan=0.0, posinf=0.0, neginf=0.0)

                    target = post_df['resNDVI'].values.astype(np.float32)
                    target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

                    # 计算样本的权重
                    sample_weights = day_weights.copy()
                    weights_list.append(sample_weights)

                    pre_ndvi_list.append(pre_ndvi)
                    eefdr_factors_list.append(eefdr_features)
                    noneefdr_factors_list.append(noneefdr_features)
                    target_list.append(target)
                    mask_list.append(mask)
                    per_sample_features_list.append(per_sample_features)

                except json.JSONDecodeError:
                    total_skipped += 1
                    continue
                except KeyError:
                    total_skipped += 1
                    continue
                except ValueError:
                    total_skipped += 1
                    continue

    print_with_timestamp = lambda msg: print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}')

    print_with_timestamp(f"Total processed samples: {len(pre_ndvi_list)}")
    print_with_timestamp(f"Total skipped samples: {total_skipped}")

    pre_ndvi_tensor = torch.tensor(pre_ndvi_list, dtype=torch.float32)  # [N, 3]
    eefdr_factors_tensor = torch.tensor(eefdr_factors_list, dtype=torch.float32)  # [N, 28, 13]
    noneefdr_factors_tensor = torch.tensor(noneefdr_factors_list, dtype=torch.float32)  # [N, 28, 10]
    target_tensor = torch.tensor(target_list, dtype=torch.float32)  # [N, 28]
    mask_tensor = torch.tensor(mask_list, dtype=torch.float32)  # [N, 28]
    per_sample_features_tensor = torch.tensor(per_sample_features_list, dtype=torch.float32)  # [N, 2]
    weights_tensor = torch.tensor(weights_list, dtype=torch.float32)  # [N, 28]

    print_with_timestamp("Standardizing features...")
    # 对eefdr_factors进行标准化
    eefdr_all_factors = eefdr_factors_tensor.view(-1, eefdr_factors_tensor.shape[-1]).numpy()
    eefdr_scaler = StandardScaler()
    eefdr_scaler.fit(eefdr_all_factors)
    eefdr_factors_scaled = eefdr_scaler.transform(
        eefdr_factors_tensor.view(-1, eefdr_factors_tensor.shape[-1]).numpy()).reshape(
        eefdr_factors_tensor.shape)
    eefdr_factors_tensor = torch.tensor(eefdr_factors_scaled, dtype=torch.float32)

    # 对noneefdr_factors进行标准化
    noneefdr_all_factors = noneefdr_factors_tensor.view(-1, noneefdr_factors_tensor.shape[-1]).numpy()
    noneefdr_scaler = StandardScaler()
    noneefdr_scaler.fit(noneefdr_all_factors)
    noneefdr_factors_scaled = noneefdr_scaler.transform(
        noneefdr_factors_tensor.view(-1, noneefdr_factors_tensor.shape[-1]).numpy()).reshape(
        noneefdr_factors_tensor.shape)
    noneefdr_factors_tensor = torch.tensor(noneefdr_factors_scaled, dtype=torch.float32)

    # 对per_sample_features进行标准化
    all_per_sample = per_sample_features_tensor.numpy()
    per_sample_scaler = StandardScaler()
    per_sample_scaler.fit(all_per_sample)
    per_sample_scaled = per_sample_scaler.transform(all_per_sample)
    per_sample_features_tensor = torch.tensor(per_sample_scaled, dtype=torch.float32)

    # 验证张量中没有 NaN 或 Inf
    tensors = [pre_ndvi_tensor, eefdr_factors_tensor, noneefdr_factors_tensor,
               target_tensor, mask_tensor, per_sample_features_tensor, weights_tensor]
    tensor_names = ['pre_ndvi_tensor', 'eefdr_factors_tensor', 'noneefdr_factors_tensor',
                    'target_tensor', 'mask_tensor', 'per_sample_features_tensor', 'weights_tensor']
    for tensor, name in zip(tensors, tensor_names):
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            raise ValueError(f"{name} contains NaN or Inf values.")

    print_with_timestamp("Saving preprocessed data...")
    torch.save({
        'pre_ndvi': pre_ndvi_tensor,
        'per_sample_features': per_sample_features_tensor,
        'eefdr_factors': eefdr_factors_tensor,
        'noneefdr_factors': noneefdr_factors_tensor,
        'target': target_tensor,
        'mask': mask_tensor,
        'weights': weights_tensor
    }, output_file)
    print_with_timestamp(f"Preprocessed data saved to {output_file}")

    print_with_timestamp("Saving scalers and LUCC encoder...")
    joblib.dump({
        'eefdr_scaler': eefdr_scaler,
        'noneefdr_scaler': noneefdr_scaler,
        'per_sample_scaler': per_sample_scaler,
        'lucc_encoder': le
    }, scaler_file)
    print_with_timestamp(f"Scalers and encoder saved to {scaler_file}")


if __name__ == '__main__':
    data_dir = 'data/rainfall_events/NDVI'
    output_file = 'preprocessed_data_with_prior.pt'
    scaler_file = 'scaler_prior.joblib'
    preprocess_data_with_prior(data_dir, output_file, scaler_file)
