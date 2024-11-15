# generate_factor_correlation_new.py

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import joblib
import os
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr  # 保留用于计算相关系数
# 移除多重比较校正工具
# from statsmodels.stats.multitest import multipletests  # 多重比较校正工具

# 导入必要的工具包
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 设置日志记录
logging.basicConfig(
    filename='generate_factor_correlation_new_14d.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device: {device}')

# 定义自定义 Dataset 类
class PreprocessedRainEventsDataset(Dataset):
    def __init__(self, pre_ndvi, per_sample_features, multi_factors, target, mask):
        self.pre_ndvi = pre_ndvi
        self.per_sample_features = per_sample_features
        self.multi_factors = multi_factors
        self.target = target
        self.mask = mask

    def __len__(self):
        return self.pre_ndvi.size(0)

    def __getitem__(self, idx):
        pre_sample_features = torch.cat([self.pre_ndvi[idx], self.per_sample_features[idx]], dim=0)  # [5]
        return pre_sample_features, self.multi_factors[idx], self.target[idx], self.mask[idx]

# 定义 Bi-LSTM 模型架构（仅用于加载模型权重）
class BiLSTMModel(torch.nn.Module):
    def __init__(self, pre_sample_features_size=5, input_size=13, hidden_size=128, num_layers=2, output_size=1, dropout=0.2):
        super(BiLSTMModel, self).__init__()
        self.pre_sample_fc = torch.nn.Linear(pre_sample_features_size, hidden_size)
        self.lstm = torch.nn.LSTM(input_size + hidden_size, hidden_size, num_layers=num_layers,
                                  batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_size * 2, output_size)  # 双向 LSTM，隐藏状态乘以2

    def forward(self, pre_sample_features, multi_factors):
        pre_sample_feat = self.pre_sample_fc(pre_sample_features)  # [batch_size, hidden_size]
        pre_sample_feat = torch.relu(pre_sample_feat)  # 应用 ReLU 激活函数
        pre_sample_feat = pre_sample_feat.unsqueeze(1).repeat(1, multi_factors.size(1), 1)  # [batch_size, 28, hidden_size]
        combined_input = torch.cat((multi_factors, pre_sample_feat), dim=2)  # [batch_size, 28, 13 + hidden_size]
        lstm_out, _ = self.lstm(combined_input)  # [batch_size, 28, hidden_size*2]
        output = self.fc(lstm_out)  # [batch_size, 28, 1]
        output = output.squeeze(-1)  # [batch_size, 28]
        return output

# 定义权重初始化函数
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                torch.nn.init.constant_(param, 0)

# 加载预处理的数据
def load_preprocessed_data(preprocessed_file, scaler_file):
    try:
        data = torch.load(preprocessed_file, map_location='cpu')
        pre_ndvi = data['pre_ndvi']
        per_sample_features = data['per_sample_features']
        multi_factors = data['multi_factors']
        target = data['target']
        mask = data['mask']
        scaler = joblib.load(scaler_file)
        logging.info(f"Loaded preprocessed data from '{preprocessed_file}' and scaler from '{scaler_file}'.")
        return pre_ndvi, per_sample_features, multi_factors, target, mask, scaler
    except Exception as e:
        logging.error(f"加载预处理数据失败: {e}")
        raise e

# 检查特征统计
def check_feature_statistics(valid_multi_factors, factor_names):
    for i, feature in enumerate(factor_names):
        unique_values = np.unique(valid_multi_factors[:, i])
        variance = np.var(valid_multi_factors[:, i])
        print(f"Feature: {feature}, Unique Values: {len(unique_values)}, Variance: {variance}")

# 定义计算贝叶斯因子的函数
def compute_bayes_factor(r, n):
    """
    计算贝叶斯因子 BF10，假设零相关的先验。
    采用近似公式：BF10 = (1 + r^2)^{- (n - 2) / 2}
    """
    if n <= 2:
        return np.nan  # 无法计算
    bf = (1 + r**2) ** (-(n - 2) / 2)
    return bf

# 定义因子相关性函数
def generate_factor_correlation_new(test_loader, factor_names, results_dir='factor_correlation_new_results_14d'):
    """
    生成新的因子相关性热图，包含颜色、圆圈大小和基于贝叶斯因子的星号标注。
    """
    all_multi_factors = []
    all_masks = []

    # 收集测试数据
    with torch.no_grad():
        for _, multi_factors_batch, _, mask_batch in test_loader:
            all_multi_factors.append(multi_factors_batch.cpu().numpy())  # [batch_size, 28, 13]
            all_masks.append(mask_batch.cpu().numpy())  # [batch_size, 28, 13]

    all_multi_factors = np.concatenate(all_multi_factors, axis=0)  # [N, 28, 13]
    all_masks = np.concatenate(all_masks, axis=0)  # [N, 28, 13]

    # 使用掩膜仅保留有效的数据
    valid_mask = all_masks.any(axis=2)  # [N, 28]
    valid_multi_factors = all_multi_factors[valid_mask]  # [num_valid, 13]
    n_samples = valid_multi_factors.shape[0]  # 有效样本量

    if valid_multi_factors.size == 0:
        logging.error("No valid data available for correlation calculation.")
        print("没有有效数据可用于相关性计算。")
        return

    # 检查特征统计
    check_feature_statistics(valid_multi_factors, factor_names)

    # 计算相关系数和 BF 矩阵
    num_factors = valid_multi_factors.shape[1]
    corr_matrix = np.zeros((num_factors, num_factors))
    bf_matrix = np.ones((num_factors, num_factors))  # 初始化为1

    for i in range(num_factors):
        for j in range(num_factors):
            if i <= j:  # 计算上三角及对角线
                if i == j:
                    corr_matrix[i, j] = 1.0
                    bf_matrix[i, j] = np.inf  # 对角线 BF 无穷大
                else:
                    r, _ = pearsonr(valid_multi_factors[:, i], valid_multi_factors[:, j])
                    corr_matrix[i, j] = r
                    bf = compute_bayes_factor(r, n_samples)
                    bf_matrix[i, j] = bf
                    corr_matrix[j, i] = r  # 对称
                    bf_matrix[j, i] = bf  # 对称

    # 创建绘图，图像大小调整为 Remote Sensing of Environment 双栏大小（17.4 cm 宽）
    plt.figure(figsize=(6.85, 6))  # 双栏宽度约 6.85 英寸
    ax = plt.gca()

    # 设置颜色映射为更加鲜明的科学配色
    cmap = sns.diverging_palette(220, 20, as_cmap=True)  # 更鲜明的红蓝配色

    # 绘制相关系数矩阵
    for i in range(num_factors):
        for j in range(num_factors):
            corr = corr_matrix[i, j]
            bf = bf_matrix[i, j]

            if i == j:
                # 对角线显示因子名称
                name = factor_names[i].replace('_', ' ')
                ax.text(j, i, name, ha='center', va='center', color='black', fontsize=11, fontfamily='Arial', fontweight='bold')
                continue

            if i < j:
                # 上三角部分绘制颜色圆圈和星号
                # 颜色
                color = cmap((corr + 1) / 2)

                # 圆圈大小限制
                max_size = 700  # 限制最大圆圈大小
                size = max_size * abs(corr)

                # 绘制圆圈
                ax.scatter(j, i, s=size, color=color, alpha=0.8, edgecolors='w', linewidth=0.5)

                # 添加贝叶斯因子标注，基于 BF
                if np.isinf(bf):
                    sig = '∞'
                elif bf < 1:
                    sig = ''
                elif 1 <= bf < 3:
                    sig = '*'
                elif 3 <= bf < 10:
                    sig = '***'
                else:  # bf >= 10
                    sig = '**********'

                if sig:
                    ax.text(j, i, sig, ha='center', va='center', color='black', fontsize=11, fontfamily='Arial')
            else:
                # 下三角部分显示相关系数数值
                corr_text = f"{corr:.2f}"
                ax.text(j, i, corr_text, ha='center', va='center', color='black', fontsize=11, fontfamily='Arial')

    # 设置轴范围
    ax.set_xlim(-0.5, num_factors - 0.5)
    ax.set_ylim(num_factors - 0.5, -0.5)

    # 移除 x 和 y 轴的刻度标签
    ax.set_xticks([])
    ax.set_yticks([])

    # 添加细虚线分隔格子
    for i in range(num_factors + 1):
        ax.axhline(i - 0.5, color='gray', linestyle='--', linewidth=0.5)
        ax.axvline(i - 0.5, color='gray', linestyle='--', linewidth=0.5)

    # 添加颜色条但不显示其标题
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])

    # 使用 make_axes_locatable 创建独立的轴来放置 colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(sm, cax=cax)
    cbar.ax.tick_params(labelsize=11)  # 设置 colorbar 的刻度字体大小为 Arial 11pts

    # 设置 colorbar 的刻度为 [-1, -0.5, 0, 0.5, 1]
    ticks = [-1, -0.5, 0, 0.5, 1]
    cbar.set_ticks(ticks)

    plt.rcParams.update({'font.size': 11, 'font.family': 'Arial'})

    # 调整布局
    plt.tight_layout()

    # 保存图像
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'factor_correlation_new_heatmap_14d.png'), dpi=300)
    plt.close()

    logging.info("Saved new factor correlation heatmap with Bayes Factors and advanced visualization.")
    print(f"新的因子相关性热图已保存至 '{results_dir}' 目录。")

# 主函数
def main():
    # 文件路径
    preprocessed_file = 'preprocessed_data_with_mask_14d.pt'
    scaler_file = 'scaler_14d.joblib'
    test_indices_file = 'test_indices_14d.npy'

    # 定义因子名称，替换下划线为空格并使用缩写
    factor_names = [
        'TEMP',  # temperature
        'DPT',   # dewpoint
        'SP',    # pressure
        'WS',    # wind_speed
        'TPS',   # total_precip_sum
        'PM10',  # pm10
        'PSD',   # pm10_pm25
        'CPS',   # current_pre
        'LON',   # lon
        'LAT',   # lat
        'LCC',   # Land cover change
        'DOS',   # day of year sin
        'DOC'    # day of year cos
    ]

    # 加载数据
    try:
        pre_ndvi, per_sample_features, multi_factors, target, mask, scaler = load_preprocessed_data(preprocessed_file, scaler_file)
    except Exception as e:
        logging.error(f"加载预处理数据失败: {e}")
        return

    # 创建数据集
    dataset = PreprocessedRainEventsDataset(pre_ndvi, per_sample_features, multi_factors, target, mask)

    # 加载测试集索引
    try:
        test_indices = np.load(test_indices_file)
        test_indices = test_indices.tolist()
        test_subset = Subset(dataset, test_indices)
        logging.info(f"Loaded test set indices from '{test_indices_file}'.")
    except FileNotFoundError:
        logging.error(f"Test indices file '{test_indices_file}' not found. 请确保它存在并且路径正确。")
        return
    except Exception as e:
        logging.error(f"加载测试集索引时出错: {e}")
        return

    # 创建 DataLoader
    batch_size = 64
    try:
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
        logging.info("Created DataLoader for test dataset.")
    except Exception as e:
        logging.error(f"创建 DataLoader 失败: {e}")
        return

    # 实例化模型（仅用于加载权重）
    hidden_size = 128
    num_layers = 2
    input_size = multi_factors.size(2)
    model = BiLSTMModel(pre_sample_features_size=5, input_size=input_size,
                       hidden_size=hidden_size, num_layers=num_layers).to(device)
    logging.info("Initialized BiLSTMModel architecture:")
    logging.info(model)

    # 初始化权重
    model.apply(init_weights)
    logging.info("Applied weight initialization to the model.")

    # 加载模型权重
    try:
        model.load_state_dict(torch.load('best_model_14d.pth', map_location=device), strict=True)
        logging.info("Loaded the best model from 'best_model_14d.pth'.")
    except FileNotFoundError:
        logging.error("Best model file 'best_model_14d.pth' not found. 请检查模型是否在训练过程中成功保存。")
        return
    except Exception as e:
        logging.error(f"加载最佳模型时出错: {e}")
        return

    # 生成新的因子相关性热图
    try:
        generate_factor_correlation_new(test_loader, factor_names, results_dir='factor_correlation_new_results')
    except Exception as e:
        logging.error(f"生成新的因子相关性热图时出错: {e}")

if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()
