# evaluate_model_with_prior.py

import torch
import torch.nn as nn
from torch.amp import autocast
import numpy as np
import joblib
import os
import json
import logging
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import torch.utils.data as data_utils
from matplotlib.colors import LogNorm

# 设置日志记录
logging.basicConfig(
    filename='evaluate_model_with_prior.log',
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


# 1. 加载预处理的数据
def load_preprocessed_data(preprocessed_file, scaler_file):
    try:
        data = torch.load(preprocessed_file, map_location='cpu')
        pre_ndvi = data['pre_ndvi']  # [N, 3]
        per_sample_features = data['per_sample_features']  # [N, 2]
        eefdr_factors = data['eefdr_factors']  # [N, 28, 13]
        noneefdr_factors = data['noneefdr_factors']  # [N, 28, 10]
        target = data['target']  # [N, 28]
        mask = data['mask']  # [N, 28]
        weights = data['weights']  # [N, 28]
        scaler = joblib.load(scaler_file)
        logging.info(f"Loaded preprocessed data from '{preprocessed_file}' and scaler from '{scaler_file}'.")
        return pre_ndvi, per_sample_features, eefdr_factors, noneefdr_factors, target, mask, weights, scaler
    except Exception as e:
        logging.error(f"Error loading preprocessed data: {e}")
        raise


# 2. 定义Dataset
class PreprocessedRainEventsDataset(Dataset):
    def __init__(self, pre_ndvi, per_sample_features, eefdr_factors, noneefdr_factors, target, mask, weights):
        self.pre_ndvi = pre_ndvi
        self.per_sample_features = per_sample_features
        self.eefdr_factors = eefdr_factors
        self.noneefdr_factors = noneefdr_factors
        self.target = target
        self.mask = mask
        self.weights = weights

    def __len__(self):
        return self.pre_ndvi.size(0)

    def __getitem__(self, idx):
        pre_sample_features = torch.cat([self.pre_ndvi[idx], self.per_sample_features[idx]], dim=0)  # [5]
        return pre_sample_features, self.eefdr_factors[idx], self.noneefdr_factors[idx], self.target[idx], self.mask[
            idx], self.weights[idx]


# 3. 定义Bi-LSTM模型架构
class BiLSTMModelWithPrior(nn.Module):
    def __init__(self, pre_sample_features_size=5, eefdr_input_size=13, noneefdr_input_size=10,
                 hidden_size=128, num_layers=2, output_size=1, dropout=0.2):
        super(BiLSTMModelWithPrior, self).__init__()
        self.pre_sample_fc = nn.Linear(pre_sample_features_size, hidden_size)
        self.eefdr_lstm = nn.LSTM(eefdr_input_size + hidden_size, hidden_size, num_layers=num_layers,
                                  batch_first=True, bidirectional=True, dropout=dropout)
        self.noneefdr_lstm = nn.LSTM(noneefdr_input_size + hidden_size, hidden_size, num_layers=num_layers,
                                     batch_first=True, bidirectional=True, dropout=dropout)
        self.eefdr_fc = nn.Linear(hidden_size * 2, output_size)
        self.noneefdr_fc = nn.Linear(hidden_size * 2, output_size)
        self.C = nn.Parameter(torch.tensor(0.0))  # 可学习参数C

    def forward(self, pre_sample_features, eefdr_factors, noneefdr_factors):
        # 处理 pre_sample_features，通过全连接层
        pre_feat = self.pre_sample_fc(pre_sample_features)  # [batch_size, hidden_size]
        pre_feat = torch.relu(pre_feat)
        seq_len = eefdr_factors.size(1)
        pre_feat_seq_eefdr = pre_feat.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, hidden_size]
        pre_feat_seq_noneefdr = pre_feat.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, hidden_size]

        # f_EEFDR
        input_eefdr = torch.cat((eefdr_factors, pre_feat_seq_eefdr), dim=2)  # [batch_size, seq_len, 13+hidden_size]
        lstm_out_eefdr, _ = self.eefdr_lstm(input_eefdr)  # [batch_size, seq_len, hidden_size*2]
        f_E = self.eefdr_fc(lstm_out_eefdr).squeeze(-1)  # [batch_size, seq_len]

        # f_Non_EEFDR
        input_noneefdr = torch.cat((noneefdr_factors, pre_feat_seq_noneefdr),
                                   dim=2)  # [batch_size, seq_len, 10+hidden_size]
        lstm_out_noneefdr, _ = self.noneefdr_lstm(input_noneefdr)  # [batch_size, seq_len, hidden_size*2]
        f_NE = self.noneefdr_fc(lstm_out_noneefdr).squeeze(-1)  # [batch_size, seq_len]

        # resNDVI = f_E + f_NE + C
        resNDVI = f_E + f_NE + self.C  # [batch_size, seq_len]

        return resNDVI, f_E, f_NE


# 初始化权重函数
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)


# 4. 定义损失函数和优化器
def get_criterion_optimizer_scheduler(model):
    def weighted_masked_mse_loss(predictions, targets, mask, weights):
        """
        计算加权掩膜的均方误差损失。
        """
        # mask: [batch_size, seq_len]
        # weights: [batch_size, seq_len]
        mse = (predictions - targets) ** 2  # [batch_size, seq_len]
        mse = mse * mask * weights  # 应用掩膜和权重
        loss = mse.sum() / (mask * weights).sum()  # 计算平均损失
        return loss

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    logging.info("Initialized weighted_masked_mse_loss as the loss function.")
    logging.info("Initialized AdamW optimizer with lr=1e-4 and weight_decay=1e-5.")
    logging.info("Initialized StepLR scheduler with step_size=10 and gamma=0.1.")
    return weighted_masked_mse_loss, optimizer, scheduler


# 5. 训练与验证函数
def train_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, patience=5):
    best_val_loss = np.inf
    patience_counter = 0

    train_losses = []
    val_losses = []

    scaler_amp = GradScaler()

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_train_loss = 0.0

        for batch_idx, (pre_sample_batch, eefdr_factors_batch, noneefdr_factors_batch,
                        target_batch, mask_batch, weights_batch) in enumerate(train_loader):
            pre_sample_batch = pre_sample_batch.to(device, non_blocking=True)
            eefdr_factors_batch = eefdr_factors_batch.to(device, non_blocking=True)
            noneefdr_factors_batch = noneefdr_factors_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            mask_batch = mask_batch.to(device, non_blocking=True)
            weights_batch = weights_batch.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast(device_type=device.type):
                outputs, _, _ = model(pre_sample_batch, eefdr_factors_batch, noneefdr_factors_batch)
                loss = criterion(outputs, target_batch, mask_batch, weights_batch)

            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"Loss is NaN or Inf for batch {batch_idx} during training. Skipping this batch.")
                continue

            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()

            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)

        # 验证
        model.eval()
        epoch_val_loss = 0.0
        val_nan_inf_count = 0
        all_val_preds = []
        all_val_targets = []
        all_val_masks = []

        with torch.no_grad():
            for batch_idx, (pre_sample_batch, eefdr_factors_batch, noneefdr_factors_batch,
                            target_batch, mask_batch, weights_batch) in enumerate(val_loader):
                pre_sample_batch = pre_sample_batch.to(device, non_blocking=True)
                eefdr_factors_batch = eefdr_factors_batch.to(device, non_blocking=True)
                noneefdr_factors_batch = noneefdr_factors_batch.to(device, non_blocking=True)
                target_batch = target_batch.to(device, non_blocking=True)
                mask_batch = mask_batch.to(device, non_blocking=True)
                weights_batch = weights_batch.to(device, non_blocking=True)

                with autocast(device_type=device.type):
                    outputs, _, _ = model(pre_sample_batch, eefdr_factors_batch, noneefdr_factors_batch)
                    loss = criterion(outputs, target_batch, mask_batch, weights_batch)

                if torch.isnan(loss) or torch.isinf(loss):
                    val_nan_inf_count += 1
                    logging.warning(f"Loss is NaN or Inf for batch {batch_idx} during validation. Skipping this batch.")
                    continue

                epoch_val_loss += loss.item()
                all_val_preds.append(outputs.cpu().numpy())
                all_val_targets.append(target_batch.cpu().numpy())
                all_val_masks.append(mask_batch.cpu().numpy())

        epoch_val_loss /= len(val_loader)
        val_losses.append(epoch_val_loss)

        if all_val_preds and all_val_targets and all_val_masks:
            all_val_preds_np = np.concatenate(all_val_preds, axis=0)  # [N, seq_len]
            all_val_targets_np = np.concatenate(all_val_targets, axis=0)  # [N, seq_len]
            all_val_masks_np = np.concatenate(all_val_masks, axis=0)  # [N, seq_len]

            valid_mask = all_val_masks_np.astype(bool)
            valid_preds = all_val_preds_np[valid_mask]
            valid_targets = all_val_targets_np[valid_mask]

            if valid_preds.size == 0 or valid_targets.size == 0:
                logging.warning("No valid data available for evaluation in this epoch.")
                mse = mae = r2 = rmse = np.nan
            else:
                mse = mean_squared_error(valid_targets, valid_preds)
                mae = mean_absolute_error(valid_targets, valid_preds)
                r2 = r2_score(valid_targets, valid_preds)
                rmse = np.sqrt(mse)
        else:
            mse = mae = r2 = rmse = np.nan

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        print(
            f'[{current_time}] Epoch {epoch}/{num_epochs}, Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}, MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.4f}, RMSE: {rmse:.6f}')
        logging.info(
            f'Epoch {epoch}: Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}, MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.4f}, RMSE: {rmse:.6f}')

        if val_nan_inf_count > 0:
            print(f'Warning: {val_nan_inf_count} batches in validation contain NaN or Inf values.')
            logging.warning(f'{val_nan_inf_count} batches in validation contain NaN or Inf values.')

        # 检查是否为最佳模型
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'best_model_prior.pth')
            patience_counter = 0
            logging.info(f"Epoch {epoch}: Best model saved with Val Loss: {epoch_val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping triggered.')
                logging.info("Early stopping triggered.")
                break

        scheduler.step()

    return train_losses, val_losses


# 6. 评估函数
def evaluate(model, test_loader, results_dir='evaluation_results_with_prior'):
    model.eval()
    all_preds = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for batch_idx, (pre_sample_batch, eefdr_factors_batch, noneefdr_factors_batch, target_batch, mask_batch,
                        weights_batch) in enumerate(test_loader):
            pre_sample_batch = pre_sample_batch.to(device, non_blocking=True)
            eefdr_factors_batch = eefdr_factors_batch.to(device, non_blocking=True)
            noneefdr_factors_batch = noneefdr_factors_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            mask_batch = mask_batch.to(device, non_blocking=True)

            with autocast(device_type=device.type):
                resNDVI, _, _ = model(pre_sample_batch, eefdr_factors_batch, noneefdr_factors_batch)

            if torch.isnan(resNDVI).any() or torch.isinf(resNDVI).any():
                logging.warning(f"Some outputs in batch {batch_idx} contain NaN or Inf. Skipping these samples.")
                continue

            if torch.isnan(target_batch).any() or torch.isinf(target_batch).any():
                logging.warning(f"Some target values in batch {batch_idx} contain NaN or Inf. Skipping these samples.")
                continue

            all_preds.append(resNDVI.cpu().numpy())
            all_targets.append(target_batch.cpu().numpy())
            all_masks.append(mask_batch.cpu().numpy())

    if not all_preds or not all_targets or not all_masks:
        logging.error("No valid predictions or targets found during evaluation.")
        return np.array([]), np.array([]), None, None, None

    all_preds = np.concatenate(all_preds, axis=0)  # [N, 28]
    all_targets = np.concatenate(all_targets, axis=0)  # [N, 28]
    all_masks = np.concatenate(all_masks, axis=0)  # [N, 28]

    valid_mask = all_masks.astype(bool)
    valid_preds = all_preds[valid_mask]
    valid_targets = all_targets[valid_mask]
    n_valid = valid_preds.size

    if valid_preds.size == 0 or valid_targets.size == 0:
        logging.error("No valid data available for evaluation after applying masks.")
        return np.array([]), np.array([]), None, None, None

    # 计算评价指标
    mse = mean_squared_error(valid_targets, valid_preds)
    mae = mean_absolute_error(valid_targets, valid_preds)
    r2 = r2_score(valid_targets, valid_preds)
    rmse = np.sqrt(mse)

    logging.info(f'Test Set Evaluation Metrics - MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.4f}, RMSE: {rmse:.6f}')

    metrics = {
        'MSE': float(mse),
        'MAE': float(mae),
        'R2': float(r2),
        'RMSE': float(rmse),
        'n_valid': int(n_valid)
    }

    try:
        os.makedirs(results_dir, exist_ok=True)
        np.save(os.path.join(results_dir, 'all_preds.npy'), valid_preds)
        np.save(os.path.join(results_dir, 'all_targets.npy'), valid_targets)
        with open(os.path.join(results_dir, 'evaluation_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        logging.info("Valid predictions, targets, and evaluation metrics have been saved successfully.")
    except Exception as e:
        logging.error(f"Error saving evaluation results: {e}")

    # 打印评价指标
    print(f'Test Set Evaluation Metrics:')
    for key, value in metrics.items():
        if 'R2' in key:
            print(f'{key}: {value:.4f}')
        else:
            print(f'{key}: {value:.6f}')

    return all_preds, all_targets, None, None, metrics


# 7. 可视化函数
def visualize(train_losses, val_losses, all_preds, all_targets, all_f_E, all_f_NE):

    # 绘制训练和验证损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig('evaluation_results_with_prior/loss_curve.png', dpi=300)
    plt.close()
    logging.info("Generated training and validation loss curves.")

    # 绘制resNDVI的Hexbin散点图
    plt.figure(figsize=(10, 10))
    hb = plt.hexbin(all_targets.flatten(), all_preds.flatten(), gridsize=500, cmap='coolwarm', mincnt=1, bins='log',
                    norm=LogNorm())
    cb = plt.colorbar(hb)
    cb.set_label('log10(N)')
    plt.plot([-0.1, 0.1], [-0.1, 0.1], 'k-', label='1:1 Line')
    reg = LinearRegression()
    reg.fit(all_targets.flatten().reshape(-1, 1), all_preds.flatten())
    y_pred = reg.predict(np.array([all_targets.min(), all_targets.max()]).reshape(-1, 1))
    plt.plot([all_targets.min(), all_targets.max()], y_pred, 'r--', label='Fit Line')
    r2 = r2_score(all_targets.flatten(), all_preds.flatten())
    rmse = np.sqrt(mean_squared_error(all_targets.flatten(), all_preds.flatten()))
    plt.xlabel('True resNDVI')
    plt.ylabel('Predicted resNDVI')
    plt.title(f'Predicted vs True resNDVI\nn={len(all_preds.flatten())}, $R^2$={r2:.3f}, RMSE={rmse:.3f}')
    plt.legend()
    plt.xlim([-0.1, 0.1])
    plt.ylim([-0.1, 0.1])
    plt.savefig('evaluation_results_with_prior/hexbin_scatter_plot_resNDVI.png', dpi=300)
    plt.close()
    logging.info("Generated Hexbin scatter plot for resNDVI.")

    # 生成残差分布图
    residuals = all_preds - all_targets
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals.flatten(), bins=50, kde=True)
    plt.xlabel('Residual (Predicted - True)')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution for resNDVI')
    plt.savefig('evaluation_results_with_prior/residual_distribution_resNDVI.png', dpi=300)
    plt.close()
    logging.info("Generated residual distribution plot for resNDVI.")


# 8. 主函数
def main():
    results_dir = 'evaluation_results_with_prior'
    os.makedirs(results_dir, exist_ok=True)

    # 加载结果
    try:
        all_preds, all_targets, all_f_E, all_f_NE, metrics = load_results(results_dir)
        logging.info("Loaded predictions and targets successfully.")
    except FileNotFoundError as e:
        logging.error(e)
        return
    except Exception as e:
        logging.error(f"Unexpected error while loading results: {e}")
        return

    # 生成可视化图像
    visualize(train_losses=[], val_losses=[], all_preds=all_preds, all_targets=all_targets, all_f_E=None, all_f_NE=None)
    logging.info("Visualization completed successfully.")


def load_results(results_dir='evaluation_results_with_prior'):
    """加载预测结果、真实值和评估指标。"""
    all_preds_path = os.path.join(results_dir, 'all_preds.npy')
    all_targets_path = os.path.join(results_dir, 'all_targets.npy')
    metrics_path = os.path.join(results_dir, 'evaluation_metrics.json')

    if not os.path.exists(all_preds_path) or not os.path.exists(all_targets_path):
        error_msg = "Predictions or targets files not found in the specified directory."
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

    all_preds = np.load(all_preds_path)
    all_targets = np.load(all_targets_path)

    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    else:
        metrics = None

    return all_preds, all_targets, None, None, metrics


if __name__ == '__main__':
    main()
