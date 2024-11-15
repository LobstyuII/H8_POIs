# preprocess_data_with_mask.py

import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from datetime import datetime


def preprocess_data(data_dir, output_file='preprocessed_data_with_mask_14d.pt',
                   scaler_file='scaler_14d.joblib', le_file='lucc_encoder.joblib',
                   max_post_days=28, pre_ndvi_days=3):
    """
    预处理 rainfall_events 数据集并分别保存张量和 scaler。

    参数:
        data_dir (str): 包含 rainfall_events CSV 文件的目录。
        output_file (str): 保存预处理后的张量数据的路径。
        scaler_file (str): 保存 StandardScaler 和 LabelEncoder 对象的路径。
        le_file (str): 保存 LUCC 标签编码器的路径。
        max_post_days (int): 考虑降雨后最多的天数。
        pre_ndvi_days (int): 降雨前 NDVI 残差的天数。
    """
    pre_ndvi_list = []
    multi_factors_list = []
    target_list = []
    mask_list = []
    per_sample_features_list = []

    multi_factors_columns = [
        'temperature', 'dewpoint_temperature', 'surface_pressure',
        'wind_speed', 'total_precipitation_sum', 'PM10',
        'PM10_PM25', 'current_precipitation', 'lat', 'lon'
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
                    post_df[multi_factors_columns] = post_df[multi_factors_columns].fillna(0.0)
                    post_df['LUCC_encoded'] = le.transform(post_df['LUCC'].fillna('Unclassified'))

                    post_df['date'] = pd.to_datetime(post_df['date'], format='%Y%m%d', errors='coerce')
                    post_df['day_of_year'] = post_df['date'].dt.dayofyear.fillna(1).astype(int)
                    post_df['day_of_year_sin'] = np.sin(2 * np.pi * post_df['day_of_year'] / 365.0)
                    post_df['day_of_year_cos'] = np.cos(2 * np.pi * post_df['day_of_year'] / 365.0)

                    post_df = post_df.fillna(0.0)

                    final_multi_factors = post_df[multi_factors_columns].values.astype(np.float32)
                    lucc_encoded = post_df['LUCC_encoded'].values.reshape(-1, 1).astype(np.float32)
                    day_sin = post_df['day_of_year_sin'].values.reshape(-1, 1).astype(np.float32)
                    day_cos = post_df['day_of_year_cos'].values.reshape(-1, 1).astype(np.float32)

                    combined_factors = np.hstack([
                        final_multi_factors,
                        lucc_encoded,
                        day_sin,
                        day_cos
                    ])  # [28,13]

                    # 创建时间步级别的 mask，1 表示有效，0 表示填充
                    time_mask = np.zeros((max_post_days, 13), dtype=np.float32)
                    time_mask[:actual_length, :] = 1.0  # [28,13]

                    # 创建特征级别的 mask，1 表示有效，0 表示缺失
                    feature_mask = (~post_df[multi_factors_columns].isna()).astype(np.float32).values  # [28,10]
                    additional_mask = np.ones((max_post_days, 3), dtype=np.float32)  # [28,3]
                    combined_feature_mask = np.hstack([feature_mask, additional_mask])  # [28,13]

                    # 综合掩膜，只有时间步有效且特征有效的部分为1，其余为0
                    final_mask = time_mask * combined_feature_mask  # [28,13]

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

                    pre_ndvi_list.append(pre_ndvi)
                    multi_factors_list.append(combined_factors)
                    target_list.append(target)
                    mask_list.append(final_mask)
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
    multi_factors_tensor = torch.tensor(multi_factors_list, dtype=torch.float32)  # [N, 28, 13]
    target_tensor = torch.tensor(target_list, dtype=torch.float32)  # [N, 28]
    mask_tensor = torch.tensor(mask_list, dtype=torch.float32)  # [N, 28, 13]
    per_sample_features_tensor = torch.tensor(per_sample_features_list, dtype=torch.float32)  # [N, 2]

    print_with_timestamp("Standardizing multi_factors and per_sample_features...")
    all_factors = multi_factors_tensor.view(-1, multi_factors_tensor.shape[-1]).numpy()
    scaler = StandardScaler()
    scaler.fit(all_factors)
    multi_factors_scaled = scaler.transform(
        multi_factors_tensor.view(-1, multi_factors_tensor.shape[-1]).numpy()).reshape(
        multi_factors_tensor.shape)
    multi_factors_tensor = torch.tensor(multi_factors_scaled, dtype=torch.float32)

    all_per_sample = per_sample_features_tensor.numpy()
    scaler_sample = StandardScaler()
    scaler_sample.fit(all_per_sample)
    per_sample_scaled = scaler_sample.transform(all_per_sample)
    per_sample_features_tensor = torch.tensor(per_sample_scaled, dtype=torch.float32)

    # 验证张量中没有 NaN 或 Inf
    if torch.isnan(pre_ndvi_tensor).any() or torch.isinf(pre_ndvi_tensor).any():
        raise ValueError("pre_ndvi_tensor contains NaN or Inf values.")
    if torch.isnan(multi_factors_tensor).any() or torch.isinf(multi_factors_tensor).any():
        raise ValueError("multi_factors_tensor contains NaN or Inf values.")
    if torch.isnan(target_tensor).any() or torch.isinf(target_tensor).any():
        raise ValueError("target_tensor contains NaN or Inf values.")
    if torch.isnan(mask_tensor).any() or torch.isinf(mask_tensor).any():
        raise ValueError("mask_tensor contains NaN or Inf values.")
    if torch.isnan(per_sample_features_tensor).any() or torch.isinf(per_sample_features_tensor).any():
        raise ValueError("per_sample_features_tensor contains NaN or Inf values.")

    print_with_timestamp("Saving preprocessed data...")
    torch.save({
        'pre_ndvi': pre_ndvi_tensor,
        'per_sample_features': per_sample_features_tensor,
        'multi_factors': multi_factors_tensor,
        'target': target_tensor,
        'mask': mask_tensor
    }, output_file)
    print_with_timestamp(f"Preprocessed data saved to {output_file}")

    print_with_timestamp("Saving scaler and LUCC encoder...")
    joblib.dump({
        'multi_factors_scaler': scaler,
        'per_sample_scaler': scaler_sample,
        'lucc_encoder': le
    }, scaler_file)
    print_with_timestamp(f"Scaler and encoder saved to {scaler_file}")


if __name__ == '__main__':
    data_dir = 'data/rainfall_events_NDVI_14d'
    output_file = 'preprocessed_data_with_mask_14d.pt'
    scaler_file = 'scaler_14d.joblib'
    preprocess_data(data_dir, output_file, scaler_file)
