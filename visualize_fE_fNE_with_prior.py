# plot_residuals_f_E_f_NE.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import json
import logging
import seaborn as sns

logging.basicConfig(
    filename='plot_residuals_f_E_f_NE.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def load_results(results_dir='evaluation_results_with_prior'):
    """加载预测结果、真实值和评估指标。"""
    all_f_E_path = os.path.join(results_dir, 'all_f_E.npy')
    all_f_NE_path = os.path.join(results_dir, 'all_f_NE.npy')
    all_targets_path = os.path.join(results_dir, 'all_targets.npy')
    metrics_path = os.path.join(results_dir, 'evaluation_metrics.json')

    required_files = [all_f_E_path, all_f_NE_path, all_targets_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            error_msg = f"Required file '{file_path}' not found in '{results_dir}'."
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)

    # 加载数据
    all_f_E = np.load(all_f_E_path)
    all_f_NE = np.load(all_f_NE_path)
    all_targets = np.load(all_targets_path)

    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    else:
        metrics = None
        logging.warning(f"Metrics file '{metrics_path}' not found.")

    logging.info("Loaded all_f_E, all_f_NE, and all_targets successfully.")
    return all_f_E, all_f_NE, all_targets, metrics

def plot_residual_distribution(residuals, xlabel, save_path):
    """绘制残差分布柱状图。"""
    # 设置字体为 Arial，大小为 11 pts
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 11

    plt.figure(figsize=(2.5, 2))

    # 绘制柱状图
    sns.histplot(residuals, bins=50, kde=False, color='blue')

    # 设置 X 轴范围
    plt.xlim([-0.1, 0.1])

    # 设置 X 轴标签
    plt.xlabel(xlabel)

    # 去除 Y 轴标签
    plt.ylabel('')  # 去除 Y 轴标签

    # 去除标题
    plt.title('')  # 去除标题

    ax = plt.gca()

    # 设置 y 轴刻度为每5k一个刻度
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10000))

    # 将 y 轴刻度除以10000，并格式化为一位小数
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/10000:.1f}'))

    # 在图表左上角添加 '×10^4' 标注
    ax.text(0.02, 0.98, r'$\times 10^{4}$', transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info(f"Saved residual distribution plot to '{save_path}'.")

def main():
    results_dir = 'evaluation_results_with_prior'
    os.makedirs(results_dir, exist_ok=True)

    # 加载结果
    try:
        all_f_E, all_f_NE, all_targets, metrics = load_results(results_dir)
    except FileNotFoundError as e:
        logging.error(e)
        print(e)
        return
    except Exception as e:
        logging.error(f"Unexpected error while loading results: {e}")
        print(e)
        return

    # 计算残差
    residuals_E = all_f_E - all_targets
    residuals_NE = all_f_NE - all_targets

    # 绘制 residual_distribution_f_E
    save_path_E = os.path.join(results_dir, 'residual_distribution_f_E.png')
    plot_residual_distribution(residuals_E.flatten(), xlabel='f_E', save_path=save_path_E)

    # 绘制 residual_distribution_f_NE
    save_path_NE = os.path.join(results_dir, 'residual_distribution_f_NE.png')
    plot_residual_distribution(residuals_NE.flatten(), xlabel='f_NE', save_path=save_path_NE)

    print("Residual distribution plots for f_E and f_NE have been generated successfully.")
    logging.info("Residual distribution plots for f_E and f_NE have been generated successfully.")

if __name__ == '__main__':
    main()
