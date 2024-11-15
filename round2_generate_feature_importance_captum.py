import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import joblib
import os
import logging
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 定义因子名称，替换下划线为空格并使用缩写
factor_names = [
    'TEMP',  # temperature
    'DPT',  # dewpoint
    'SP',  # pressure
    'WS',  # wind speed
    'TPS',  # total precip sum
    'PM10',  # pm10
    'PSD',  # pm10 pm25
    'CPS',  # current pre
    'LON',  # lon
    'LAT',  # lat
    'LCC',  # Land cover change
    'DOS',  # day of year sin
    'DOC'  # day of year cos
]

# 设置日志记录
logging.basicConfig(
    filename='generate_feature_importance_captum_corrected_14d.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'使用设备: {device}')


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


# 定义 Bi-LSTM 模型架构
class BiLSTMModel(nn.Module):
    def __init__(self, pre_sample_features_size=5, input_size=13, hidden_size=128, num_layers=2, output_size=1,
                 dropout=0.2):
        super(BiLSTMModel, self).__init__()
        self.pre_sample_fc = nn.Linear(pre_sample_features_size, hidden_size)
        self.lstm = nn.LSTM(input_size + hidden_size, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, pre_sample_features, multi_factors):
        pre_sample_feat = self.pre_sample_fc(pre_sample_features)
        pre_sample_feat = torch.relu(pre_sample_feat)
        pre_sample_feat = pre_sample_feat.unsqueeze(1).repeat(1, multi_factors.size(1),
                                                              1)  # [batch_size, 28, hidden_size]
        combined_input = torch.cat((multi_factors, pre_sample_feat), dim=2)  # [batch_size, 28, 13 + hidden_size]
        lstm_out, _ = self.lstm(combined_input)  # [batch_size, 28, hidden_size * 2]
        output = self.fc(lstm_out)  # [batch_size, 28, 1]
        output = output.squeeze(-1)  # [batch_size, 28]
        return output


# 加载预处理的数据
def load_preprocessed_data(preprocessed_file, scaler_file):
    try:
        # 加载数据
        data = torch.load(preprocessed_file, map_location='cpu')
        pre_ndvi = data['pre_ndvi']
        per_sample_features = data['per_sample_features']
        multi_factors = data['multi_factors']
        target = data['target']
        mask = data['mask']

        # 加载缩放器
        scaler = joblib.load(scaler_file)
        logging.info(f"从 '{preprocessed_file}' 加载预处理数据，并从 '{scaler_file}' 加载缩放器。")
        return pre_ndvi, per_sample_features, multi_factors, target, mask, scaler
    except Exception as e:
        logging.error(f"加载预处理数据失败: {e}")
        raise e


# 定义特征重要性可视化函数
def generate_feature_importance_captum_viz(mean_attributions_multi, factor_names, total_attribution,
                                           results_dir='feature_importance_results_14d'):
    """
    生成特征重要性图，并显示归一化解释力（百分比）和总特征贡献（Total Attribution）。

    参数:
    - mean_attributions_multi: 各因子的平均绝对重要性
    - factor_names: 因子名称列表
    - total_attribution: 总特征贡献（非归一化）
    - results_dir: 结果保存目录
    """
    # 归一化每个特征的重要性，计算百分比
    normalized_attributions = mean_attributions_multi / np.sum(mean_attributions_multi) * 100

    # 按重要性排序多因子特征
    sorted_indices = np.argsort(normalized_attributions)[::-1]  # 从大到小排序
    sorted_attributions = normalized_attributions[sorted_indices]
    sorted_factor_names = [factor_names[i] for i in sorted_indices]

    # 交换 'PSD' 和 'DOC'
    psd_index = sorted_factor_names.index('PSD')
    doc_index = sorted_factor_names.index('DOC')
    sorted_factor_names[psd_index], sorted_factor_names[doc_index] = sorted_factor_names[doc_index], sorted_factor_names[psd_index]

    # 交换 'PM10' 和 'DOS'
    pm10_index = sorted_factor_names.index('PM10')
    dos_index = sorted_factor_names.index('DOS')
    sorted_factor_names[pm10_index], sorted_factor_names[dos_index] = sorted_factor_names[dos_index], sorted_factor_names[pm10_index]

    # 设置图幅尺寸为 70% 高度
    plt.figure(figsize=(6, 3))

    # 定义y轴位置
    y_positions = np.arange(len(sorted_attributions))

    # 绘制每个特征的水平虚线
    plt.hlines(y=y_positions, xmin=0, xmax=sorted_attributions, colors='gray', linestyles='dashed', alpha=0.7)

    # 绘制特征重要性点（黑色空心圆）
    plt.scatter(sorted_attributions, y_positions, facecolors='none', edgecolors='black')

    # 设置y轴刻度标签为特征名称，并设置字体为Arial，大小为11
    plt.yticks(y_positions, sorted_factor_names, fontfamily='Arial', fontsize=11)

    # 设置x轴标签为归一化解释力（百分比）
    plt.xlabel("Normalized Feature Importance (%)")
    plt.xlim(left=0)

    # 反转y轴，使高重要性在顶部
    plt.gca().invert_yaxis()

    # 添加图例，显示Total Attribution，并设置字体为Arial，大小为11
    plt.scatter([], [], facecolors='none', edgecolors='black', label=f"Total Attribution: {total_attribution:.6f}")
    plt.legend(loc='lower right', frameon=False, prop={'family': 'Arial', 'size': 11})

    plt.tight_layout()

    # 保存图片
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'toc_feature_importance_14d.png'), dpi=300)
    plt.close()

    logging.info("保存TOC特征重要性图。")


# 定义生成特征重要性的函数
def generate_feature_importance_captum(model, test_loader, sample_size=100, results_dir='feature_importance_results_14d'):
    """
    使用Captum的集成梯度方法生成特征重要性可视化。

    参数:
    - model: 训练好的模型
    - test_loader: 测试集的DataLoader
    - sample_size: 用于计算重要性的样本数量
    - results_dir: 结果保存目录
    """
    model.eval()  # 初始设置为评估模式
    all_pre_sample_features = []
    all_multi_factors = []
    all_targets = []

    # 收集测试数据
    with torch.no_grad():
        for pre_sample_batch, multi_factors_batch, target_batch, mask_batch in test_loader:
            all_pre_sample_features.append(pre_sample_batch.cpu())  # [batch_size, 5]
            all_multi_factors.append(multi_factors_batch.cpu())  # [batch_size, 28, 13]
            all_targets.append(target_batch.cpu())  # [batch_size]

    all_pre_sample_features = torch.cat(all_pre_sample_features, dim=0)  # [N, 5]
    all_multi_factors = torch.cat(all_multi_factors, dim=0)  # [N, 28, 13]
    all_targets = torch.cat(all_targets, dim=0)  # [N]

    # 随机选择样本
    N = all_pre_sample_features.size(0)
    if N < sample_size:
        sample_size = N
        logging.warning(f"样本数量减少到 {sample_size}，因为有效数据有限。")

    selected_indices = np.random.choice(N, sample_size, replace=False)
    selected_pre = all_pre_sample_features[selected_indices]  # [sample_size, 5]
    selected_multi = all_multi_factors[selected_indices]  # [sample_size, 28, 13]
    selected_targets = all_targets[selected_indices]  # [sample_size]

    # 移动到设备
    selected_pre = selected_pre.to(device)
    selected_multi = selected_multi.to(device)

    # 定义集成梯度方法
    ig = IntegratedGradients(model)

    # 定义前向函数，返回预测的均值
    def forward_func(pre_sample, multi_factors):
        outputs = model(pre_sample, multi_factors)  # [batch_size, 28]
        output_mean = outputs.mean(dim=1)  # [batch_size]
        return output_mean

    # 计算基线（全零）
    baseline_pre = torch.zeros_like(selected_pre).to(device)
    baseline_multi = torch.zeros_like(selected_multi).to(device)

    # 临时切换模型到训练模式，并禁用Dropout
    model.train()  # 切换到训练模式以允许反向传播
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()  # 禁用Dropout

    # 计算集成梯度
    try:
        # 由于模型输出是单一的标量，target 可以设置为0
        attributions_pre, attributions_multi = ig.attribute(
            inputs=(selected_pre, selected_multi),
            baselines=(baseline_pre, baseline_multi),
            target=0,  # 指定目标为第0个输出
            return_convergence_delta=False
        )
        logging.info("成功计算集成梯度属性。")
    except Exception as e:
        logging.error(f"计算集成梯度时出错: {e}")
        # 切换回评估模式
        model.eval()
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        return

    # 切换回评估模式
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

    # 将属性移动到CPU并转为numpy
    attributions_multi = attributions_multi.cpu().detach().numpy()  # [sample_size, 28, 13]

    # 计算平均绝对属性
    mean_attributions_multi = np.mean(np.abs(attributions_multi), axis=(0, 1))  # [13]

    # 计算总特征贡献（Total Attribution）
    total_attribution = np.sum(mean_attributions_multi)

    # 调用新的特征重要性可视化函数
    generate_feature_importance_captum_viz(mean_attributions_multi, factor_names, total_attribution, results_dir)


# 主函数
def main():
    # 文件路径
    preprocessed_file = 'preprocessed_data_with_mask_14d.pt'
    scaler_file = 'scaler_14d.joblib'
    test_indices_file = 'test_indices_14d.npy'

    # 加载数据
    try:
        pre_ndvi, per_sample_features, multi_factors, target, mask, scaler = load_preprocessed_data(preprocessed_file,
                                                                                                    scaler_file)
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
        logging.info(f"从 '{test_indices_file}' 加载测试集索引。")
    except FileNotFoundError:
        logging.error(f"测试集索引文件 '{test_indices_file}' 未找到。请确保它存在并且路径正确。")
        return
    except Exception as e:
        logging.error(f"加载测试集索引时出错: {e}")
        return

    # 创建 DataLoader
    batch_size = 64
    try:
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4,
                                 pin_memory=True)
        logging.info("创建测试数据集的 DataLoader。")
    except Exception as e:
        logging.error(f"创建 DataLoader 失败: {e}")
        return

    # 实例化模型
    hidden_size = 128
    num_layers = 2
    input_size = multi_factors.size(2)
    model = BiLSTMModel(pre_sample_features_size=5, input_size=input_size,
                        hidden_size=hidden_size, num_layers=num_layers).to(device)
    logging.info("初始化 BiLSTMModel 架构：")
    logging.info(model)

    # 加载模型权重
    try:
        model.load_state_dict(torch.load('best_model_14d.pth', map_location=device), strict=True)
        logging.info("从 'best_model_14d.pth' 加载最佳模型。")
    except FileNotFoundError:
        logging.error("最佳模型文件 'best_model_14d.pth' 未找到。请检查模型是否在训练过程中成功保存。")
        return
    except Exception as e:
        logging.error(f"加载最佳模型时出错: {e}")
        return

    # 生成特征重要性
    try:
        generate_feature_importance_captum(model, test_loader, sample_size=100, results_dir='feature_importance_results_14d')
    except Exception as e:
        logging.error(f"生成特征重要性时出错: {e}")


if __name__ == '__main__':
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()
