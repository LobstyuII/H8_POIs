import os
import glob
import json
import logging
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 设置日志
logging.basicConfig(
    filename='f_E_prediction.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 1. 定义模型架构（与训练时一致）
class BiLSTMModelWithPrior(nn.Module):
    def __init__(self, pre_sample_features_size=5, eefdr_input_size=13, noneefdr_input_size=10, hidden_size=128,
                 num_layers=2, output_size=1, dropout=0.2):
        super(BiLSTMModelWithPrior, self).__init__()
        self.pre_sample_features_size = pre_sample_features_size
        self.eefdr_input_size = eefdr_input_size
        self.noneefdr_input_size = noneefdr_input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # 预处理 pre_sample_features，线性变换
        self.pre_sample_fc = nn.Linear(pre_sample_features_size, hidden_size)

        # f(EEFDR)的双向LSTM
        self.eefdr_lstm = nn.LSTM(eefdr_input_size + hidden_size, hidden_size, num_layers=num_layers, batch_first=True,
                                  bidirectional=True, dropout=dropout)

        # f(Non_EEFDR)的双向LSTM
        self.noneefdr_lstm = nn.LSTM(noneefdr_input_size + hidden_size, hidden_size, num_layers=num_layers,
                                     batch_first=True, bidirectional=True, dropout=dropout)

        # 全连接层，将 LSTM 输出映射到输出维度
        self.eefdr_fc = nn.Linear(hidden_size * 2, output_size)
        self.noneefdr_fc = nn.Linear(hidden_size * 2, output_size)

        # C为全局平均值，可作为模型的参数学习
        self.C = nn.Parameter(torch.tensor(0.0))

    def forward(self, pre_sample_features, eefdr_factors, noneefdr_factors):
        # 处理 pre_sample_features，通过全连接层
        pre_sample_feat = self.pre_sample_fc(pre_sample_features)  # [batch_size, hidden_size]
        pre_sample_feat = torch.relu(pre_sample_feat)

        # 将 pre_sample_feat 扩展到序列长度
        seq_len = eefdr_factors.size(1)
        pre_sample_feat_seq_eefdr = pre_sample_feat.unsqueeze(1).repeat(1, seq_len,
                                                                        1)  # [batch_size, seq_len, hidden_size]
        pre_sample_feat_seq_noneefdr = pre_sample_feat.unsqueeze(1).repeat(1, seq_len,
                                                                           1)  # [batch_size, seq_len, hidden_size]

        # f(EEFDR)部分
        eefdr_input = torch.cat((eefdr_factors, pre_sample_feat_seq_eefdr),
                                dim=2)  # [batch_size, seq_len, 13+hidden_size]
        eefdr_output, _ = self.eefdr_lstm(eefdr_input)  # [batch_size, seq_len, hidden_size*2]
        f_E = self.eefdr_fc(eefdr_output).squeeze(-1)  # [batch_size, seq_len]

        # f(Non_EEFDR)部分
        noneefdr_input = torch.cat((noneefdr_factors, pre_sample_feat_seq_noneefdr),
                                   dim=2)  # [batch_size, seq_len, 10+hidden_size]
        noneefdr_output, _ = self.noneefdr_lstm(noneefdr_input)  # [batch_size, seq_len, hidden_size*2]
        f_NE = self.noneefdr_fc(noneefdr_output).squeeze(-1)  # [batch_size, seq_len]

        # resNDVI = f_E + f_NE + C
        output = f_E + f_NE + self.C  # [batch_size, seq_len]
        return output, f_E, f_NE


# 2. 定义Dataset类用于推理
class StationDataset(Dataset):
    def __init__(self, data_dir, scaler_dict, feature_columns_pre_ndvi, feature_columns_per_sample,
                 feature_columns_eefdr, feature_columns_noneefdr):
        """
        Args:
            data_dir (str): 存放所有站点_events.csv文件的目录。
            scaler_dict (dict): 包含不同特征集的Scaler对象的字典。
            feature_columns_pre_ndvi (list): pre_ndvi的列名。
            feature_columns_per_sample (list): per_sample_features的列名。
            feature_columns_eefdr (list): eefdr_factors的列名。
            feature_columns_noneefdr (list): noneefdr_factors的列名。
        """
        self.data_dir = data_dir
        self.scaler_pre_sample = scaler_dict.get('pre_sample_features', None)
        self.scaler_eefdr = scaler_dict.get('eefdr_factors', None)
        self.scaler_noneefdr = scaler_dict.get('noneefdr_factors', None)

        # 获取所有_events.csv文件路径
        self.file_paths = glob.glob(os.path.join(data_dir, '*_events.csv'))
        self.file_paths.sort()  # 确保顺序一致

        self.feature_columns_pre_ndvi = feature_columns_pre_ndvi
        self.feature_columns_per_sample = feature_columns_per_sample
        self.feature_columns_eefdr = feature_columns_eefdr
        self.feature_columns_noneefdr = feature_columns_noneefdr

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        返回：
            pre_sample_features: Tensor [5]
            eefdr_factors: Tensor [28, 13]
            noneefdr_factors: Tensor [28, 10]
            station_id: str
            event_dates: list of dates corresponding to the sequence
        """
        file_path = self.file_paths[idx]
        station_id = os.path.basename(file_path).split('_')[0]

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            logging.error(f"读取文件失败: {file_path}, 错误: {e}")
            # 返回全零的张量和空列表
            pre_sample_features = torch.zeros(len(self.feature_columns_pre_ndvi) + len(self.feature_columns_per_sample))
            eefdr_factors = torch.zeros(28, 13)
            noneefdr_factors = torch.zeros(28, 10)
            event_dates = []
            return pre_sample_features, eefdr_factors, noneefdr_factors, station_id, event_dates

        # 提取pre_ndvi特征
        pre_ndvi = df[self.feature_columns_pre_ndvi].values  # [num_events, 3]

        # 提取per_sample_features
        per_sample_features = df[self.feature_columns_per_sample].values  # [num_events, 2]

        # 提取eefdr_factors和noneefdr_factors
        # 假设'eefdr_factors'和'noneefdr_factors'存储在'post_data'列的JSON中
        post_data_str = df['post_data'].values  # [num_events]
        eefdr_factors = []
        noneefdr_factors = []
        event_dates = []
        for post_data_json in post_data_str:
            try:
                # 处理双重引号
                post_data = json.loads(post_data_json.replace('""', '"'))
                # eefdr和noneefdr特征提取
                eefdr = []
                noneefdr = []
                dates = []
                for day in post_data:
                    # 根据实际特征名称提取
                    # 请根据实际数据调整以下特征名称和顺序
                    eefdr.append([
                        day.get("temperature", 0),
                        day.get("dewpoint_temperature", 0),
                        day.get("surface_pressure", 0),
                        day.get("wind_speed", 0),
                        day.get("total_precipitation_sum", 0),
                        day.get("PM10", 0),
                        day.get("PM10_PM25", 0),
                        day.get("lon", 0),
                        day.get("lat", 0),
                        day.get("current_precipitation", 0),
                        day.get("average_pre_residual", 0),
                        day.get("dry_period", 0),
                        # 如果有更多特征，请继续添加，直到13个
                        day.get("additional_feature", 0)  # 假设有一个额外特征
                    ])
                    noneefdr.append([
                        day.get("wind_speed", 0),  # 示例特征，请根据实际情况调整
                        day.get("surface_pressure", 0),
                        day.get("temperature", 0),
                        day.get("dewpoint_temperature", 0),
                        day.get("PM10", 0),
                        day.get("PM10_PM25", 0),
                        day.get("current_precipitation", 0),
                        day.get("lon", 0),
                        day.get("lat", 0),
                        day.get("total_precipitation_sum", 0)
                        # 添加更多特征直到10个
                    ])
                    dates.append(day.get("date", ""))
                # 填充不足28天的数据
                while len(eefdr) < 28:
                    eefdr.append([0] * 13)
                    noneefdr.append([0] * 10)
                    dates.append("")
                # 截断多余的天数
                eefdr = eefdr[:28]
                noneefdr = noneefdr[:28]
                event_dates = dates[:28]
            except Exception as e:
                logging.error(f"解析post_data失败: {post_data_json}, 错误: {e}")
                eefdr = [[0] * 13 for _ in range(28)]
                noneefdr = [[0] * 10 for _ in range(28)]
                event_dates = [""] * 28

            eefdr_factors.append(eefdr)
            noneefdr_factors.append(noneefdr)

        eefdr_factors = np.array(eefdr_factors)  # [num_events, 28, 13]
        noneefdr_factors = np.array(noneefdr_factors)  # [num_events, 28, 10]

        # 假设每个_events.csv文件对应多个事件，取平均或按需处理
        # 这里假设每个文件只包含一个事件
        pre_ndvi = pre_ndvi[0] if len(pre_ndvi) > 0 else np.zeros(len(self.feature_columns_pre_ndvi))
        per_sample_features = per_sample_features[0] if len(per_sample_features) > 0 else np.zeros(
            len(self.feature_columns_per_sample))
        eefdr_factors = eefdr_factors[0] if len(eefdr_factors) > 0 else np.zeros((28, 13))
        noneefdr_factors = noneefdr_factors[0] if len(noneefdr_factors) > 0 else np.zeros((28, 10))
        event_dates = event_dates[:28] if len(event_dates) >= 28 else [""] * 28

        # 合并pre_ndvi和per_sample_features
        pre_sample_features = np.concatenate([pre_ndvi, per_sample_features], axis=0)  # [5]

        # 标准化
        if self.scaler_pre_sample:
            pre_sample_features = self.scaler_pre_sample.transform(pre_sample_features.reshape(1, -1)).flatten()
        else:
            logging.warning("No scaler found for pre_sample_features. Skipping scaling.")

        if self.scaler_eefdr:
            eefdr_factors = self.scaler_eefdr.transform(eefdr_factors.reshape(-1, eefdr_factors.shape[-1])).reshape(
                eefdr_factors.shape)
        else:
            logging.warning("No scaler found for eefdr_factors. Skipping scaling.")

        if self.scaler_noneefdr:
            noneefdr_factors = self.scaler_noneefdr.transform(
                noneefdr_factors.reshape(-1, noneefdr_factors.shape[-1])).reshape(noneefdr_factors.shape)
        else:
            logging.warning("No scaler found for noneefdr_factors. Skipping scaling.")

        # 转换为Tensor
        pre_sample_features = torch.tensor(pre_sample_features, dtype=torch.float32)
        eefdr_factors = torch.tensor(eefdr_factors, dtype=torch.float32)
        noneefdr_factors = torch.tensor(noneefdr_factors, dtype=torch.float32)

        # 添加断言确保 event_dates 长度为28
        assert len(event_dates) == 28, f"event_dates length is {len(event_dates)}, expected 28."

        return pre_sample_features, eefdr_factors, noneefdr_factors, station_id, event_dates


# 3. 定义自定义的 collate_fn
def custom_collate(batch):
    pre_sample_features, eefdr_factors, noneefdr_factors, station_ids, event_dates = zip(*batch)
    pre_sample_features = torch.stack(pre_sample_features)
    eefdr_factors = torch.stack(eefdr_factors)
    noneefdr_factors = torch.stack(noneefdr_factors)
    station_ids = list(station_ids)
    event_dates = list(event_dates)  # list of lists
    return pre_sample_features, eefdr_factors, noneefdr_factors, station_ids, event_dates


# 4. 加载Scaler
def load_scaler(scaler_path):
    try:
        scaler_dict = joblib.load(scaler_path)
        if not isinstance(scaler_dict, dict):
            logging.error("Scaler文件不是一个字典。请确保Scaler是以字典形式保存的。")
            raise ValueError("Scaler文件格式不正确。")
        logging.info(f"成功加载Scaler: {scaler_path}")
        return scaler_dict
    except Exception as e:
        logging.error(f"加载Scaler失败: {e}")
        raise e


# 5. 加载训练好的模型
def load_model(model_path):
    try:
        # 根据训练时的参数初始化模型
        model = BiLSTMModelWithPrior(
            pre_sample_features_size=5,
            eefdr_input_size=13,
            noneefdr_input_size=10,
            hidden_size=128,
            num_layers=2,
            output_size=1,
            dropout=0.2
        )
        # 设置 weights_only=True 来应对 FutureWarning
        # 如果 torch.load 支持 weights_only 参数，则设置为 True
        # 否则，使用默认值并忽略警告
        if hasattr(torch.load, 'weights_only'):
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
        else:
            state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        logging.info(f"成功加载模型: {model_path}")
        return model
    except Exception as e:
        logging.error(f"加载模型失败: {e}")
        raise e


# 6. 主函数
def main():
    data_dir = 'data/rainfall_events/NDVI'
    scaler_path = 'scaler_prior.joblib'
    model_path = 'best_model_prior.pth'

    # 定义特征列名（根据实际数据调整）
    feature_columns_pre_ndvi = ['pre_NDVI_residual_day1', 'pre_NDVI_residual_day2', 'pre_NDVI_residual_day3']
    feature_columns_per_sample = ['current_precipitation', 'average_pre_residual']
    feature_columns_eefdr = [
        "temperature",
        "dewpoint_temperature",
        "surface_pressure",
        "wind_speed",
        "total_precipitation_sum",
        "PM10",
        "PM10_PM25",
        "lon",
        "lat",
        "current_precipitation",
        "average_pre_residual",
        "dry_period",
        "additional_feature"
    ]
    feature_columns_noneefdr = [
        "wind_speed",
        "surface_pressure",
        "temperature",
        "dewpoint_temperature",
        "PM10",
        "PM10_PM25",
        "current_precipitation",
        "lon",
        "lat",
        "total_precipitation_sum"
        # 添加更多特征直到10个
    ]

    # 加载Scaler
    scaler_dict = load_scaler(scaler_path)

    # 创建Dataset和DataLoader，使用自定义的 collate_fn
    dataset = StationDataset(
        data_dir=data_dir,
        scaler_dict=scaler_dict,
        feature_columns_pre_ndvi=feature_columns_pre_ndvi,
        feature_columns_per_sample=feature_columns_per_sample,
        feature_columns_eefdr=feature_columns_eefdr,
        feature_columns_noneefdr=feature_columns_noneefdr
    )
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, collate_fn=custom_collate)

    # 加载模型
    model = load_model(model_path)

    # 准备保存结果的DataFrame
    all_predictions = []

    # 遍历DataLoader进行预测
    logging.info("开始进行f_E预测...")
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Predicting f_E")):
        pre_sample_features, eefdr_factors, noneefdr_factors, station_ids, event_dates = batch
        pre_sample_features = pre_sample_features.to(device)
        eefdr_factors = eefdr_factors.to(device)
        noneefdr_factors = noneefdr_factors.to(device)

        # Debugging: log the lengths
        logging.info(
            f"Batch {batch_idx}: station_ids length = {len(station_ids)}, event_dates length = {len(event_dates)}")
        print(f"Batch {batch_idx}: station_ids length = {len(station_ids)}, event_dates length = {len(event_dates)}")

        # Ensure station_ids and event_dates have the same length
        if len(station_ids) != len(event_dates):
            logging.error(
                f"Batch {batch_idx}: station_ids length ({len(station_ids)}) != event_dates length ({len(event_dates)}). Skipping this batch.")
            print(
                f"Batch {batch_idx}: station_ids length ({len(station_ids)}) != event_dates length ({len(event_dates)}). Skipping this batch.")
            continue

        with torch.no_grad():
            _, f_E, _ = model(pre_sample_features, eefdr_factors, noneefdr_factors)  # f_E: [batch_size, seq_len]

        f_E = f_E.cpu().numpy()  # 转换为numpy数组

        for i in range(len(station_ids)):
            station_id = station_ids[i]
            dates = event_dates[i]
            # Ensure dates is a list of length 28
            if not isinstance(dates, list):
                logging.error(f"Batch {batch_idx}, Sample {i}: event_dates is not a list. Skipping this sample.")
                print(f"Batch {batch_idx}, Sample {i}: event_dates is not a list. Skipping this sample.")
                continue
            if len(dates) != 28:
                logging.error(
                    f"Batch {batch_idx}, Sample {i}: event_dates length is {len(dates)}, expected 28. Skipping this sample.")
                print(
                    f"Batch {batch_idx}, Sample {i}: event_dates length is {len(dates)}, expected 28. Skipping this sample.")
                continue

            for j, date in enumerate(dates):
                if j >= f_E.shape[1]:
                    logging.warning(f"Batch {batch_idx}, Sample {i}, j={j} 超出 f_E 的范围 {f_E.shape[1]}，跳过此日期。")
                    continue  # 跳过超出范围的索引
                if date == "":
                    continue  # 跳过无效日期
                prediction = f_E[i, j]
                all_predictions.append({
                    'station': station_id,
                    'date': date,
                    'f_E_prediction': prediction
                })

    if not all_predictions:
        logging.error("没有任何预测结果被生成。请检查数据和模型。")
        print("没有任何预测结果被生成。请检查数据和模型。")
        return

    # 创建结果的DataFrame
    df_predictions = pd.DataFrame(all_predictions)
    df_predictions['date'] = pd.to_datetime(df_predictions['date'], format='%Y%m%d')

    # 保存所有站点的预测结果到总CSV
    total_csv_path = 'f_E_predictions_all_stations.csv'
    df_predictions.to_csv(total_csv_path, index=False)
    logging.info(f"所有站点的f_E预测结果已保存到: {total_csv_path}")
    print(f"所有站点的f_E预测结果已保存到: {total_csv_path}")

    # 另外，为每个站点单独保存CSV
    stations = df_predictions['station'].unique()
    output_dir = 'f_E_predictions_per_station'
    os.makedirs(output_dir, exist_ok=True)
    for station in tqdm(stations, desc="Saving per-station CSVs"):
        df_station = df_predictions[df_predictions['station'] == station]
        station_csv_path = os.path.join(output_dir, f"{station}_f_E_predictions.csv")
        df_station.to_csv(station_csv_path, index=False)

    # 计算每日平均f_E值
    df_daily_avg = df_predictions.groupby(['station', 'date'])['f_E_prediction'].mean().reset_index()
    df_daily_avg.rename(columns={'f_E_prediction': 'f_E_daily_avg'}, inplace=True)

    # 保存每日平均值到CSV
    daily_avg_csv_path = 'f_E_daily_average_all_stations.csv'
    df_daily_avg.to_csv(daily_avg_csv_path, index=False)
    logging.info(f"每日平均f_E值已保存到: {daily_avg_csv_path}")
    print(f"每日平均f_E值已保存到: {daily_avg_csv_path}")

    plt.figure(figsize=(15, 10))
    for station in stations:
        df_station = df_daily_avg[df_daily_avg['station'] == station]
        plt.plot(df_station['date'], df_station['f_E_daily_avg'], label=station, alpha=0.5)
    plt.xlabel('Date')
    plt.ylabel('Daily Average f_E Prediction')
    plt.title('Daily Average f_E Predictions for All Stations')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=2)
    plt.tight_layout()
    plt.savefig('f_E_daily_average_all_stations.png', dpi=300)
    plt.close()
    logging.info("已生成所有站点的每日平均f_E预测值时间序列图: f_E_daily_average_all_stations.png")
    print("已生成所有站点的每日平均f_E预测值时间序列图: f_E_daily_average_all_stations.png")

    sample_station = stations[0] if len(stations) > 0 else None
    if sample_station:
        df_sample = df_daily_avg[df_daily_avg['station'] == sample_station]
        plt.figure(figsize=(15, 6))
        sns.lineplot(data=df_sample, x='date', y='f_E_daily_avg')
        plt.xlabel('Date')
        plt.ylabel('Daily Average f_E Prediction')
        plt.title(f'Daily Average f_E Predictions for Station {sample_station}')
        plt.tight_layout()
        sample_plot_path = f'f_E_daily_average_{sample_station}.png'
        plt.savefig(sample_plot_path, dpi=300)
        plt.close()
        logging.info(f"已生成站点 {sample_station} 的每日平均f_E预测值时间序列图: {sample_plot_path}")
        print(f"已生成站点 {sample_station} 的每日平均f_E预测值时间序列图: {sample_plot_path}")

    logging.info("f_E预测及可视化完成。")
    print("f_E预测及可视化完成。")


if __name__ == '__main__':
    main()
