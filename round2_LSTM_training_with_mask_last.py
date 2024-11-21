# LSTM_training_with_mask.py

import torch  
import torch.nn as nn  
from torch.utils.data import Dataset, DataLoader, random_split  
from torch.amp import GradScaler, autocast  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  
import joblib  
import os  
import logging  
from datetime import datetime  
import torch.utils.data as data_utils

logging.basicConfig(
    filename='LSTM_training_with_mask_14d.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# 设置随机种子以确保结果可重复
torch.manual_seed(42)
np.random.seed(42)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 加载预处理的数据
def load_preprocessed_data(preprocessed_file, scaler_file):
    """
    加载预处理的数据和 scaler。

    参数:
        preprocessed_file (str): 预处理数据的文件路径。
        scaler_file (str): scaler 文件的路径。

    返回:
        pre_ndvi (torch.Tensor): 预先处理的 NDVI 数据，形状为 [N, 3]。
        per_sample_features (torch.Tensor): 每个样本的特征，形状为 [N, 2]。
        multi_factors (torch.Tensor): 多因子数据，形状为 [N, 28, 13]。
        target (torch.Tensor): 目标值，形状为 [N, 28]。
        mask (torch.Tensor): 掩膜数据，形状为 [N, 28, 13]。
        scaler (dict): scaler 对象，用于数据标准化。
    """
    try:
        # data = torch.load(preprocessed_file)
        data = torch.load(preprocessed_file, map_location='cpu')
        pre_ndvi = data['pre_ndvi']  # 获取预先处理的 NDVI 数据，形状为 [N, 3]
        per_sample_features = data['per_sample_features']  # 获取每个样本的特征，形状为 [N, 2]
        multi_factors = data['multi_factors']  # 获取多因子数据，形状为 [N, 28, 13]
        target = data['target']  # 获取目标值，形状为 [N, 28]
        mask = data['mask']  # 获取掩膜数据，形状为 [N, 28, 13]

        # 加载 scaler 对象，用于数据标准化
        scaler = joblib.load(scaler_file)

        return pre_ndvi, per_sample_features, multi_factors, target, mask, scaler
    except Exception as e:
        logging.error(f"加载预处理数据失败: {e}")
        raise e

# 2. 定义自定义 Dataset 类
class PreprocessedRainEventsDataset(Dataset):
    def __init__(self, pre_ndvi, per_sample_features, multi_factors, target, mask):
        """
        初始化自定义数据集。

        参数:
            pre_ndvi (torch.Tensor): 预先处理的 NDVI 数据，形状为 [N, 3]。
            per_sample_features (torch.Tensor): 每个样本的特征，形状为 [N, 2]。
            multi_factors (torch.Tensor): 多因子数据，形状为 [N, 28, 13]。
            target (torch.Tensor): 目标值，形状为 [N, 28]。
            mask (torch.Tensor): 掩膜数据，形状为 [N, 28, 13]。
        """
        self.pre_ndvi = pre_ndvi
        self.per_sample_features = per_sample_features
        self.multi_factors = multi_factors
        self.target = target
        self.mask = mask

    def __len__(self):
        """
        返回数据集的大小。

        返回:
            int: 数据集中的样本数量。
        """
        return self.pre_ndvi.size(0)

    def __getitem__(self, idx):
        """
        获取指定索引的样本。

        参数:
            idx (int): 样本索引。

        返回:
            tuple: 包含 pre_sample_features、multi_factors、target 和 mask 的元组。
        """
        # 合并 pre_ndvi 和 per_sample_features 为 pre_sample_features，形状为 [5]
        pre_sample_features = torch.cat([self.pre_ndvi[idx], self.per_sample_features[idx]], dim=0)  # [5]
        return pre_sample_features, self.multi_factors[idx], self.target[idx], self.mask[idx]

# 3. 定义 Bi-LSTM 模型架构
class BiLSTMModel(nn.Module):
    def __init__(self, pre_sample_features_size=5, input_size=13, hidden_size=128, num_layers=2, output_size=1, dropout=0.2):
        """
        初始化 Bi-LSTM 模型。

        参数:
            pre_sample_features_size (int): 预先处理特征的数量（默认5）。
            input_size (int): 多因子特征的数量（默认13）。
            hidden_size (int): LSTM 隐藏单元的数量（默认128）。
            num_layers (int): LSTM 层数（默认2）。
            output_size (int): 输出特征的数量（默认1，用于预测 resNDVI）。
            dropout (float): dropout 率（默认0.2）。
        """
        super(BiLSTMModel, self).__init__()  # 调用父类的构造函数
        self.pre_sample_features_size = pre_sample_features_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # 预处理 pre_sample_features，线性变换
        self.pre_sample_fc = nn.Linear(pre_sample_features_size, hidden_size)

        # 双向 LSTM 层，输入维度为 multi_factors + pre_sample_features
        self.lstm = nn.LSTM(input_size + hidden_size, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)

        # 全连接层，将 LSTM 输出映射到输出维度
        self.fc = nn.Linear(hidden_size * 2, output_size)  # 双向 LSTM，隐藏状态乘以2

    def forward(self, pre_sample_features, multi_factors):
        """
        前向传播函数。

        参数:
            pre_sample_features (torch.Tensor): 预先处理的特征，形状为 [batch_size, 5]。
            multi_factors (torch.Tensor): 多因子数据，形状为 [batch_size, 28, 13]。

        返回:
            torch.Tensor: 模型输出，形状为 [batch_size, 28]。
        """
        # 处理 pre_sample_features，通过全连接层
        pre_sample_feat = self.pre_sample_fc(pre_sample_features)  # [batch_size, hidden_size]
        pre_sample_feat = torch.relu(pre_sample_feat)  # 应用 ReLU 激活函数

        # 将 pre_sample_feat 扩展到序列长度，形状为 [batch_size, 28, hidden_size]
        pre_sample_feat = pre_sample_feat.unsqueeze(1).repeat(1, multi_factors.size(1), 1)  # [batch_size, 28, hidden_size]

        # 将 pre_sample_feat 与 multi_factors 拼接，形成 LSTM 的输入
        combined_input = torch.cat((multi_factors, pre_sample_feat), dim=2)  # [batch_size, 28, 13 + hidden_size]

        # 通过 LSTM 层
        lstm_out, _ = self.lstm(combined_input)  # [batch_size, 28, hidden_size*2]

        # 通过全连接层
        output = self.fc(lstm_out)  # [batch_size, 28, 1]

        # 去除最后一个维度，形状为 [batch_size, 28]
        output = output.squeeze(-1)  # [batch_size, 28]

        return output  # 返回模型输出

# 初始化权重的函数
def init_weights(m):
    """
    初始化模型的权重。

    参数:
        m (nn.Module): 模型中的模块。
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # 使用 Xavier 均匀初始化线性层的权重
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)  # 将偏置初始化为0
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)  # 使用 Xavier 均匀初始化 LSTM 的权重
            elif 'bias' in name:
                nn.init.constant_(param, 0)  # 将偏置初始化为0

# 4. 定义损失函数和优化器
def get_criterion_optimizer_scheduler(model):
    """
    获取损失函数、优化器和学习率调度器。

    参数:
        model (nn.Module): 需要训练的模型。

    返回:
        criterion (function): 自定义的掩膜损失函数。
        optimizer (torch.optim.Optimizer): 优化器。
        scheduler (torch.optim.lr_scheduler._LRScheduler): 学习率调度器。
    """
    # 使用自定义的掩膜损失函数
    def masked_mse_loss(predictions, targets, mask):
        """
        计算掩膜的均方误差损失。

        参数:
            predictions (torch.Tensor): 模型预测，形状为 [batch_size, 28]。
            targets (torch.Tensor): 真实值，形状为 [batch_size, 28]。
            mask (torch.Tensor): 掩膜，形状为 [batch_size, 28, 13]。

        返回:
            torch.Tensor: 计算得到的损失。
        """
        # 将 mask 从 [batch_size, 28, 13] 转换为 [batch_size, 28]
        mask = mask.any(dim=2).float()  # 如果某一天的任何特征是有效的，则标记为1.0

        # 计算 MSE
        mse = (predictions - targets) ** 2  # [batch_size, 28]

        # 应用掩膜
        mse = mse * mask  # 仅计算有效数据的损失

        # 计算平均损失
        return mse.sum() / mask.sum()

    # 使用 AdamW 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # 使用 StepLR 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    return masked_mse_loss, optimizer, scheduler

# 5. 训练与验证函数
def train_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, patience=5):
    """
    训练和验证模型。

    参数:
        model (nn.Module): 要训练的模型。
        train_loader (DataLoader): 训练数据加载器。
        val_loader (DataLoader): 验证数据加载器。
        criterion (function): 自定义的掩膜损失函数。
        optimizer (torch.optim.Optimizer): 优化器。
        scheduler (torch.optim.lr_scheduler._LRScheduler): 学习率调度器。
        num_epochs (int): 训练的总 epoch 数（默认50）。
        patience (int): 提前停止的耐心值（默认5）。

    返回:
        train_losses (list): 每个 epoch 的训练损失。
        val_losses (list): 每个 epoch 的验证损失。
    """
    best_val_loss = np.inf  # 初始化最佳验证损失为无穷大
    patience_counter = 0  # 提前停止计数器

    train_losses = []  # 存储每个 epoch 的训练损失
    val_losses = []  # 存储每个 epoch 的验证损失

    # 定义用于自动混合精度训练的梯度缩放器
    scaler_amp = GradScaler()

    for epoch in range(1, num_epochs + 1):
        model.train()  # 设置模型为训练模式
        epoch_train_loss = 0.0  # 初始化本 epoch 的训练损失

        for batch_idx, (pre_sample_batch, multi_factors_batch, target_batch, mask_batch) in enumerate(train_loader):
            # 将数据移动到指定设备（GPU 或 CPU）
            pre_sample_batch = pre_sample_batch.to(device, non_blocking=True)
            multi_factors_batch = multi_factors_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            mask_batch = mask_batch.to(device, non_blocking=True)

            optimizer.zero_grad()  # 清零优化器的梯度

            # 使用自动混合精度上下文管理器
            with autocast(device_type=device.type):
                outputs = model(pre_sample_batch, multi_factors_batch)  # 模型前向传播
                loss = criterion(outputs, target_batch, mask_batch)  # 计算损失

            # 检查损失是否为 NaN 或 Inf
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"Loss is NaN or Inf for batch {batch_idx} during training. Skipping this batch.")
                continue  # 跳过该批次

            scaler_amp.scale(loss).backward()  # 反向传播并缩放梯度
            scaler_amp.unscale_(optimizer)  # 反缩放梯度

            # 添加梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler_amp.step(optimizer)  # 更新模型参数
            scaler_amp.update()  # 更新缩放因子

            epoch_train_loss += loss.item()  # 累加训练损失

        epoch_train_loss /= len(train_loader)  # 计算平均训练损失
        train_losses.append(epoch_train_loss)  # 存储本 epoch 的训练损失

        # 验证阶段
        model.eval()  # 设置模型为评估模式
        epoch_val_loss = 0.0  # 初始化本 epoch 的验证损失
        val_nan_inf_count = 0  # 记录验证阶段 NaN 或 Inf 的批次数
        all_val_preds = []  # 存储验证集的所有预测值
        all_val_targets = []  # 存储验证集的所有目标值
        all_val_masks = []  # 存储验证集的所有掩膜

        with torch.no_grad():  # 不计算梯度
            for batch_idx, (pre_sample_batch, multi_factors_batch, target_batch, mask_batch) in enumerate(val_loader):
                # 将数据移动到指定设备
                pre_sample_batch = pre_sample_batch.to(device, non_blocking=True)
                multi_factors_batch = multi_factors_batch.to(device, non_blocking=True)
                target_batch = target_batch.to(device, non_blocking=True)
                mask_batch = mask_batch.to(device, non_blocking=True)

                # 使用自动混合精度上下文管理器
                with autocast(device_type=device.type):
                    outputs = model(pre_sample_batch, multi_factors_batch)  # 模型前向传播
                    loss = criterion(outputs, target_batch, mask_batch)  # 计算损失

                # 检查损失是否为 NaN 或 Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    val_nan_inf_count += 1
                    logging.warning(f"Loss is NaN or Inf for batch {batch_idx} during validation. Skipping this batch.")
                    continue  # 跳过该批次

                epoch_val_loss += loss.item()  # 累加验证损失
                all_val_preds.append(outputs.cpu().numpy())  # 将预测值移动到 CPU 并转换为 NumPy 数组
                all_val_targets.append(target_batch.cpu().numpy())  # 将目标值移动到 CPU 并转换为 NumPy 数组
                all_val_masks.append(mask_batch.cpu().numpy())  # 将掩膜移动到 CPU 并转换为 NumPy 数组

        epoch_val_loss /= len(val_loader)  # 计算平均验证损失
        val_losses.append(epoch_val_loss)  # 存储本 epoch 的验证损失

        # 将所有验证集的预测值、目标值和掩膜拼接起来
        if all_val_preds and all_val_targets and all_val_masks:
            all_val_preds_np = np.concatenate(all_val_preds, axis=0)  # [N, 28]
            all_val_targets_np = np.concatenate(all_val_targets, axis=0)  # [N, 28]
            all_val_masks_np = np.concatenate(all_val_masks, axis=0)  # [N, 28, 13]

            # 仅保留有效数据
            valid_mask = all_val_masks_np.any(axis=2)  # [N, 28]
            valid_preds = all_val_preds_np[valid_mask]  # [num_valid]
            valid_targets = all_val_targets_np[valid_mask]  # [num_valid]

            if valid_preds.size == 0 or valid_targets.size == 0:
                logging.warning("No valid data available for evaluation in this epoch.")
                mse = mae = r2 = rmse = np.nan
            else:
                # 计算评价指标
                mse = mean_squared_error(valid_targets, valid_preds)  # 均方误差
                mae = mean_absolute_error(valid_targets, valid_preds)  # 平均绝对误差
                r2 = r2_score(valid_targets, valid_preds)  # 决定系数
                rmse = np.sqrt(mse)  # 均方根误差
        else:
            mse = mae = r2 = rmse = np.nan  # 如果没有有效的预测值和目标值，设为 NaN

        # 获取当前时间戳
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 打印并记录本 epoch 的损失和评价指标，添加时间戳
        print(f'[{current_time}] Epoch {epoch}/{num_epochs}, Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}, MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.4f}, RMSE: {rmse:.6f}')
        logging.info(f'Epoch {epoch}: Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}, MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.4f}, RMSE: {rmse:.6f}')

        if val_nan_inf_count > 0:
            print(f'Warning: {val_nan_inf_count} batches in validation contain NaN or Inf values.')
            logging.warning(f'{val_nan_inf_count} batches in validation contain NaN or Inf values.')

        # 检查是否为最佳模型
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss  # 更新最佳验证损失
            torch.save(model.state_dict(), 'best_model_14d.pth')  # 保存最佳模型
            patience_counter = 0  # 重置提前停止计数器
            logging.info(f"Epoch {epoch}: Best model saved with Val Loss: {epoch_val_loss:.6f}")
        else:
            patience_counter += 1  # 增加提前停止计数器
            if patience_counter >= patience:
                print('Early stopping triggered.')  # 打印提前停止信息
                logging.info("Early stopping triggered.")  # 记录日志
                break  # 终止训练

        scheduler.step()  # 更新学习率调度器

    return train_losses, val_losses  # 返回训练和验证损失列表

# 6. 评估函数
def evaluate(model, test_loader):
    """
    评估模型在测试集上的表现。

    参数:
        model (nn.Module): 训练好的模型。
        test_loader (DataLoader): 测试数据加载器。

    返回:
        all_preds (np.ndarray): 测试集的所有预测值。
        all_targets (np.ndarray): 测试集的所有目标值。
    """
    model.eval()  # 设置模型为评估模式
    all_preds = []  # 存储所有预测值
    all_targets = []  # 存储所有目标值
    all_masks = []  # 存储所有掩膜

    with torch.no_grad():  # 不计算梯度
        for pre_sample_batch, multi_factors_batch, target_batch, mask_batch in test_loader:
            # 将数据移动到指定设备
            pre_sample_batch = pre_sample_batch.to(device, non_blocking=True)
            multi_factors_batch = multi_factors_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            mask_batch = mask_batch.to(device, non_blocking=True)

            # 使用自动混合精度上下文管理器
            with autocast(device_type=device.type):
                outputs = model(pre_sample_batch, multi_factors_batch)  # 模型前向传播

            # 检查输出是否包含 NaN 或 Inf
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                logging.warning("Some outputs in the test set contain NaN or Inf. Skipping these samples.")
                continue  # 跳过这些样本

            # 检查目标值是否包含 NaN 或 Inf
            if torch.isnan(target_batch).any() or torch.isinf(target_batch).any():
                logging.warning("Some target values in the test set contain NaN or Inf. Skipping these samples.")
                continue  # 跳过这些样本

            all_preds.append(outputs.cpu().numpy())  # 将预测值移动到 CPU 并转换为 NumPy 数组
            all_targets.append(target_batch.cpu().numpy())  # 将目标值移动到 CPU 并转换为 NumPy 数组
            all_masks.append(mask_batch.cpu().numpy())  # 将掩膜移动到 CPU 并转换为 NumPy 数组

    # 如果没有有效的预测值或目标值，记录错误并返回空数组
    if not all_preds or not all_targets or not all_masks:
        logging.error("No valid predictions or targets found during evaluation.")
        return np.array([]), np.array([])

    # 将所有预测值、目标值和掩膜拼接成一个大的 NumPy 数组
    all_preds = np.concatenate(all_preds, axis=0)  # [N, 28]
    all_targets = np.concatenate(all_targets, axis=0)  # [N, 28]
    all_masks = np.concatenate(all_masks, axis=0)  # [N, 28, 13]

    # 仅保留有效数据
    valid_mask = all_masks.any(axis=2)  # [N, 28]
    valid_preds = all_preds[valid_mask]  # [num_valid]
    valid_targets = all_targets[valid_mask]  # [num_valid]

    if valid_preds.size == 0 or valid_targets.size == 0:
        logging.error("No valid data available for evaluation after applying masks.")
        return np.array([]), np.array([])

    # 计算评价指标
    mse = mean_squared_error(valid_targets, valid_preds)  # 均方误差
    mae = mean_absolute_error(valid_targets, valid_preds)  # 平均绝对误差
    r2 = r2_score(valid_targets, valid_preds)  # 决定系数
    rmse = np.sqrt(mse)  # 均方根误差

    # 打印评价指标
    print(f'Test Set Evaluation Metrics:')
    print(f'MSE: {mse:.6f}')
    print(f'MAE: {mae:.6f}')
    print(f'R²: {r2:.4f}')
    print(f'RMSE: {rmse:.6f}')

    # 记录评价指标到日志
    logging.info(f'Test Set Evaluation Metrics - MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.4f}, RMSE: {rmse:.6f}')

    return all_preds, all_targets  # 返回所有预测值和目标值

# 7. 可视化函数
def visualize(train_losses, val_losses, all_preds, all_targets):
    """
    可视化训练和验证损失曲线、预测与实际值的关系、时间序列对比以及残差分布。

    参数:
        train_losses (list): 每个 epoch 的训练损失。
        val_losses (list): 每个 epoch 的验证损失。
        all_preds (np.ndarray): 所有预测值。
        all_targets (np.ndarray): 所有目标值。
    """
    # 7.1 绘制训练和验证损失曲线
    plt.figure(figsize=(10, 6))  # 创建一个10x6英寸的图
    plt.plot(train_losses, label='Train Loss')  # 绘制训练损失曲线
    plt.plot(val_losses, label='Validation Loss')  # 绘制验证损失曲线
    plt.xlabel('Epoch')  # x轴标签
    plt.ylabel('Loss (MSE)')  # y轴标签
    plt.title('Training and Validation Loss Curves')  # 图表标题
    plt.legend()  # 显示图例
    plt.tight_layout()  # 自动调整子图参数
    plt.savefig('loss_curve_14d.png')  # 保存图像
    plt.close()  # 关闭图像

    # 7.2 Hexbin 图：预测 NDVI 与实际 NDVI 的关系，并添加分位数线
    plt.figure(figsize=(8, 8))  # 创建一个8x8英寸的图
    hb = plt.hexbin(all_targets.flatten(), all_preds.flatten(), gridsize=50, cmap='Blues', mincnt=1)  # 绘制 Hexbin 图
    plt.colorbar(hb, label='Counts')  # 添加颜色条
    plt.plot([all_targets.min(), all_targets.max()], [all_targets.min(), all_targets.max()], 'r--', label='Ideal')  # 绘制理想对角线

    # 计算残差
    residuals = all_preds.flatten() - all_targets.flatten()

    # 计算分位数
    quantiles = [0.25, 0.5, 0.75, 0.9]
    quantile_values = np.quantile(residuals, quantiles)

    # 绘制分位数线
    for q, q_value in zip(quantiles, quantile_values):
        plt.axhline(y=q_value, color='g', linestyle='-', label=f'{int(q * 100)}th Quantile')

    plt.xlabel('Actual NDVI')  # x轴标签
    plt.ylabel('Predicted NDVI')  # y轴标签
    plt.title('Predicted vs Actual NDVI Hexbin Plot with Quantiles')  # 图表标题
    plt.legend(loc='upper left', fontsize='small')  # 显示图例，位置在左上角，字体较小
    plt.tight_layout()  # 自动调整子图参数
    plt.savefig('hexbin_scatter_plot_14d.png')  # 保存图像
    plt.close()  # 关闭图像

    # 7.3 时间序列图：随机选择的样本的实际值与预测值对比
    if len(all_targets) >= 5:
        num_samples = 5  # 如果样本数量大于等于5，选择5个样本
    else:
        num_samples = len(all_targets)  # 否则，选择所有样本
    if num_samples > 0:
        indices = np.random.choice(len(all_targets), num_samples, replace=False)  # 随机选择样本索引

        plt.figure(figsize=(12, 8))  # 创建一个12x8英寸的图
        for idx in indices:
            plt.plot(all_targets[idx], label=f'Actual Sample {idx}')  # 绘制实际值
            plt.plot(all_preds[idx], linestyle='--', label=f'Predicted Sample {idx}')  # 绘制预测值，使用虚线
        plt.xlabel('Days')  # x轴标签
        plt.ylabel('NDVI Residual')  # y轴标签
        plt.title('Time Series Prediction Comparison')  # 图表标题
        plt.legend()  # 显示图例
        plt.tight_layout()  # 自动调整子图参数
        plt.savefig('time_series_plot_14d.png')  # 保存图像
        plt.close()  # 关闭图像

    # 7.4 残差分布图
    residuals = all_preds - all_targets  # 计算残差
    plt.figure(figsize=(10, 6))  # 创建一个10x6英寸的图
    sns.histplot(residuals.flatten(), bins=50, kde=True)  # 绘制残差的直方图和核密度估计
    plt.xlabel('Residual (Predicted - Actual)')  # x轴标签
    plt.title('Residual Distribution')  # 图表标题
    plt.tight_layout()  # 自动调整子图参数
    plt.savefig('residual_distribution_14d.png')  # 保存图像
    plt.close()  # 关闭图像

    # 绘制残差与预测值的关系图
    plt.figure(figsize=(10, 6))  # 创建一个10x6英寸的图
    plt.scatter(all_preds.flatten(), residuals.flatten(), alpha=0.3)  # 绘制散点图，设置透明度为0.3
    plt.xlabel('Predicted NDVI')  # x轴标签
    plt.ylabel('Residual')  # y轴标签
    plt.title('Residuals vs Predicted NDVI')  # 图表标题
    plt.tight_layout()  # 自动调整子图参数
    plt.savefig('residual_plot_14d.png')  # 保存图像
    plt.close()  # 关闭图像

# 8. 主函数
def main():
    """
    主函数，负责加载数据、创建数据集和数据加载器、实例化模型、训练与验证、评估模型以及可视化结果。
    """
    # 加载预处理的数据
    preprocessed_file = 'preprocessed_data_with_mask_14d.pt'  # 预处理数据的文件名
    scaler_file = 'scaler_14d.joblib'  # scaler 的文件路径
    try:
        # 调用函数加载预处理的数据和 scaler
        pre_ndvi, per_sample_features, multi_factors, target, mask, scaler = load_preprocessed_data(preprocessed_file, scaler_file)
    except Exception as e:
        logging.error(f"加载预处理数据失败: {e}")  # 记录错误日志
        return  # 终止主函数

    # 创建自定义数据集
    dataset = PreprocessedRainEventsDataset(pre_ndvi, per_sample_features, multi_factors, target, mask)

    # 数据集划分，70% 训练集，15% 验证集，15% 测试集
    train_size = int(0.7 * len(dataset))  # 训练集大小
    val_size = int(0.15 * len(dataset))  # 验证集大小
    test_size = len(dataset) - train_size - val_size  # 测试集大小

    # 使用 random_split 划分数据集
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # 保存测试集的索引
    test_indices = test_dataset.indices
    np.save('test_indices_14d.npy', test_indices)
    logging.info("Saved test set indices to 'test_indices_14d.npy'.")

    # 创建 DataLoader，设置批次大小、是否打乱、是否丢弃最后一个不完整的批次、使用的工作线程数和是否固定内存
    batch_size = 64  # 批次大小
    try:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4,
                                  pin_memory=True)  # 训练数据加载器
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4,
                                pin_memory=True)  # 验证数据加载器
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4,
                                 pin_memory=True)  # 测试数据加载器
    except Exception as e:
        logging.error(f"创建 DataLoader 失败: {e}")  # 记录错误日志
        return  # 终止主函数

    # 实例化模型
    hidden_size = 128  # LSTM 隐藏单元数量
    num_layers = 2  # LSTM 层数
    input_size = multi_factors.size(2)  # 多因子特征的数量，应为13
    model = BiLSTMModel(pre_sample_features_size=5, input_size=input_size,
                       hidden_size=hidden_size, num_layers=num_layers).to(device)  # 将模型移动到设备
    print(model)  # 打印模型结构

    # 应用权重初始化
    model.apply(init_weights)  # 对模型的每个子模块应用权重初始化函数

    # 定义损失函数和优化器
    criterion, optimizer, scheduler = get_criterion_optimizer_scheduler(model)

    # 训练与验证
    try:
        train_losses, val_losses = train_validate(model, train_loader, val_loader, criterion, optimizer, scheduler,
                                                  num_epochs=50, patience=5)  # 开始训练与验证
    except Exception as e:
        logging.error(f"训练过程中出现错误: {e}")  # 记录错误日志
        return  # 终止主函数

    # 加载最佳模型并评估
    try:
        model.load_state_dict(torch.load('best_model_14d.pth', map_location=device))  # 加载最佳模型参数
        all_preds, all_targets = evaluate(model, test_loader)  # 评估模型

        if all_preds.size != 0 and all_targets.size != 0:
            # 如果评估结果不为空，则进行可视化
            visualize(train_losses, val_losses, all_preds, all_targets)
    except FileNotFoundError:
        # 如果最佳模型文件不存在，打印提示信息并记录日志
        print("Best model file 'best_model_14d.pth' not found. Please check if the model was successfully saved during training.")
        logging.error("Best model file 'best_model_14d.pth' not found.")
    except Exception as e:
        # 如果评估过程中出现其他错误，记录错误日志
        logging.error(f"评估过程中出现错误: {e}")

if __name__ == '__main__':
    import torch.multiprocessing as mp  # 导入多进程模块

    try:
        mp.freeze_support()  # 仅在Windows上需要，启用多进程支持
    except AttributeError:
        pass
    main()
