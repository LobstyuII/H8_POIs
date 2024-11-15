import ee
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tenacity import retry, stop_after_attempt, wait_fixed
from scipy.ndimage import gaussian_filter1d  # 用于平滑曲线

# 设置 Seaborn 样式
sns.set(style="whitegrid")


def GEE_authorizing():
    """
    使用服务账户授权 GEE，并设置最大重试次数为5
    """
    service_account = "lobstyu@premium-cipher-424203-d0.iam.gserviceaccount.com"
    credentials_path = 'premium-cipher-424203-d0-c6894a29d00c.json'

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
    def initialize_ee():
        credentials = ee.ServiceAccountCredentials(service_account, credentials_path)
        ee.Initialize(credentials)

    initialize_ee()


# 授权 GEE
GEE_authorizing()

# 获取中国（包括台湾!!）的国界数据
# china = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(ee.Filter.eq('ADM0_NAME', 'China'))
china = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(
    ee.Filter.inList('ADM0_NAME', ['China', 'Taiwan'])
)
geo_json = china.getInfo()
gdf_china = gpd.GeoDataFrame.from_features(geo_json['features'])

# 读取测站经纬度数据
stations = pd.read_csv('Air_stations_lon_lat.csv', header=None, names=['station', 'lon', 'lat'])

# 读取 f_E 和 f_NE 平均数据
# 假设 'f_E_daily_average_all_stations.csv' 包含 'station', 'f_E_daily_avg', 'f_NE_daily_avg'
fE_fNE = pd.read_csv('f_E_daily_average_all_stations.csv')

# 计算每个站点的总体平均 f_E 和 f_NE
fE_fNE_avg = fE_fNE.groupby('station').agg({'f_E_daily_avg': 'mean', 'f_NE_daily_avg': 'mean'}).reset_index()

# 合并测站位置和 f_E、f_NE 数据
stations_fE_fNE = pd.merge(stations, fE_fNE_avg, on='station', how='left')

# 创建几何点
geometry = [Point(xy) for xy in zip(stations_fE_fNE['lon'], stations_fE_fNE['lat'])]
gdf_stations_fE_fNE = gpd.GeoDataFrame(stations_fE_fNE, geometry=geometry)

# 定义自适应颜色映射
# f_E 的阶梯为 0.00:0.01:0.04
fE_bins = [0.00, 0.01, 0.02, 0.03, 0.04]
fE_colors = ['#66bd63', '#fdae61', '#f46d43', '#d73027']  # 红色到绿色渐变

# f_NE 的阶梯为 -0.06:0.03:0.06
fNE_bins = [-0.06, -0.03, 0.00, 0.03, 0.06]
fNE_colors = ['#d4eac7', '#a1d99b', '#74c476', '#238b45']  # 绿色渐变

# 创建 ListedColormap 和 BoundaryNorm
fE_cmap = ListedColormap(fE_colors)
fE_norm = BoundaryNorm(fE_bins, fE_cmap.N)

fNE_cmap = ListedColormap(fNE_colors)
fNE_norm = BoundaryNorm(fNE_bins, fNE_cmap.N)

# ------------------ 计算条带图所需的数据 ------------------
# 计算纬度和经度的边界
lat_min = stations_fE_fNE['lat'].min()
lat_max = stations_fE_fNE['lat'].max()
lon_min = stations_fE_fNE['lon'].min()
lon_max = stations_fE_fNE['lon'].max()

# 定义每2.5度的纬度和经度分箱
lat_bins = np.arange(np.floor(lat_min / 2.5) * 2.5, np.ceil(lat_max / 2.5) * 2.5 + 2.5, 2.5)
lon_bins = np.arange(np.floor(lon_min / 2.5) * 2.5, np.ceil(lon_max / 2.5) * 2.5 + 2.5, 2.5)

# 对站点进行分箱
stations_fE_fNE['lat_bin'] = pd.cut(stations_fE_fNE['lat'], bins=lat_bins, include_lowest=True)
stations_fE_fNE['lon_bin'] = pd.cut(stations_fE_fNE['lon'], bins=lon_bins, include_lowest=True)

# 计算每个纬度分箱的数据量和平均值
lat_group_fE = stations_fE_fNE.groupby('lat_bin').agg(
    data_amount=('station', 'count'),
    f_E_avg=('f_E_daily_avg', 'mean'),
    f_NE_avg=('f_NE_daily_avg', 'mean')
).reset_index()
# 计算纬度中心点
lat_group_fE['lat_center'] = lat_group_fE['lat_bin'].apply(lambda x: (x.left + x.right) / 2)

# 对平均值进行平滑处理，sigma=0.2
lat_group_fE['f_E_avg_smooth'] = gaussian_filter1d(lat_group_fE['f_E_avg'], sigma=0.2)
lat_group_fE['f_NE_avg_smooth'] = gaussian_filter1d(lat_group_fE['f_NE_avg'], sigma=0.2)

# 计算每个经度分箱的数据量和平均值
lon_group_fE = stations_fE_fNE.groupby('lon_bin').agg(
    data_amount=('station', 'count'),
    f_E_avg=('f_E_daily_avg', 'mean'),
    f_NE_avg=('f_NE_daily_avg', 'mean')
).reset_index()
# 计算经度中心点
lon_group_fE['lon_center'] = lon_group_fE['lon_bin'].apply(lambda x: (x.left + x.right) / 2)

# 对平均值进行平滑处理，sigma=0.2
lon_group_fE['f_E_avg_smooth'] = gaussian_filter1d(lon_group_fE['f_E_avg'], sigma=0.1)
lon_group_fE['f_NE_avg_smooth'] = gaussian_filter1d(lon_group_fE['f_NE_avg'], sigma=0.1)
# ------------------ 数据计算结束 ------------------

# 创建绘图，调整图形大小为 (14, 6)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 设置字体为 Arial 11pt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11

# 绘制中国国界，使用浅灰色填充，线条粗细为0.75
gdf_china.plot(ax=axes[0], color='lightgrey', edgecolor='black', linewidth=0.75)
gdf_china.plot(ax=axes[1], color='lightgrey', edgecolor='black', linewidth=0.75)

# 绘制测站位置，使用颜色表示 f_E 浓度，实心圆带黑色边框，线条粗细为0.75
gdf_stations_fE_fNE.plot(ax=axes[0],
                         marker='o',
                         markersize=30,
                         column='f_E_daily_avg',
                         cmap=fE_cmap,
                         norm=fE_norm,
                         edgecolor='black',
                         linewidth=0.75,
                         alpha=0.9,
                         legend=False)

# 绘制测站位置，使用颜色表示 f_NE 浓度，实心圆带黑色边框，线条粗细为0.75
gdf_stations_fE_fNE.plot(ax=axes[1],
                         marker='o',
                         markersize=30,
                         column='f_NE_daily_avg',
                         cmap=fNE_cmap,
                         norm=fNE_norm,
                         edgecolor='black',
                         linewidth=0.75,
                         alpha=0.9,
                         legend=False)

# 设置图形边界
for ax in axes:
    ax.set_xlim(stations_fE_fNE['lon'].min() - 3, stations_fE_fNE['lon'].max() + 5)
    ax.set_ylim(stations_fE_fNE['lat'].min() - 3, stations_fE_fNE['lat'].max() + 5)


# 定义刻度格式化函数
def format_degree(x, pos):
    return f"{x:.1f}°"


for ax in axes:
    ax.xaxis.set_major_locator(plt.MultipleLocator(15))
    ax.yaxis.set_major_locator(plt.MultipleLocator(15))
    ax.xaxis.set_major_formatter(FuncFormatter(format_degree))
    ax.yaxis.set_major_formatter(FuncFormatter(format_degree))
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.grid(False)
    ax.tick_params(axis='both', which='both', length=10, width=0.75)
    # 将经纬度刻度和标签移动到上方和右侧
    ax.xaxis.tick_top()
    ax.yaxis.tick_right()
    ax.xaxis.set_label_position('top')
    ax.yaxis.set_label_position('right')
    # 设置图框的线条加粗，并确保为黑色
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color('black')

    # 添加 colorbar for f_E
cax_fE = fig.add_axes([0.15, 0.05, 0.3, 0.02])  # [left, bottom, width, height]
cb_fE = plt.colorbar(plt.cm.ScalarMappable(norm=fE_norm, cmap=fE_cmap),
                     cax=cax_fE,
                     orientation='horizontal',
                     ticks=fE_bins)
cb_fE.ax.set_xticklabels([f"{b:.2f}" for b in fE_bins[:-1]] + [f"{fE_bins[-1]:.2f}"])
cb_fE.set_label('Average Daily f_E', fontsize=11, fontname='Arial')
cb_fE.ax.xaxis.set_label_position('top')
cb_fE.ax.xaxis.set_ticks_position('top')

# 添加 colorbar for f_NE
cax_fNE = fig.add_axes([0.55, 0.05, 0.3, 0.02])  # [left, bottom, width, height]
cb_fNE = plt.colorbar(plt.cm.ScalarMappable(norm=fNE_norm, cmap=fNE_cmap),
                      cax=cax_fNE,
                      orientation='horizontal',
                      ticks=fNE_bins)
cb_fNE.ax.set_xticklabels([f"{b:.2f}" for b in fNE_bins[:-1]] + [f"{fNE_bins[-1]:.2f}"])
cb_fNE.set_label('Average Daily f_NE', fontsize=11, fontname='Arial')
cb_fNE.ax.xaxis.set_label_position('top')
cb_fNE.ax.xaxis.set_ticks_position('top')

# 添加标题
axes[0].set_title('Average Daily f_E Distribution', fontsize=14, fontname='Arial')
axes[1].set_title('Average Daily f_NE Distribution', fontsize=14, fontname='Arial')

# ------------------ 绘制条带状小图 ------------------
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 对于 f_E 图
divider_fE = make_axes_locatable(axes[0])

# 左侧条带小图（纬度）
ax_left_fE = divider_fE.append_axes("left", size="20%", pad=0.1, sharey=axes[0])
ax_left_fE_twin = ax_left_fE.twiny()

# 绘制数据量柱状图
ax_left_fE.barh(lat_group_fE['lat_center'], lat_group_fE['data_amount'], height=2.0, color='gray', alpha=0.7)

# 绘制平均值折线图
ax_left_fE_twin.plot(lat_group_fE['f_E_avg_smooth'], lat_group_fE['lat_center'], color='red', linewidth=1.25)

# 移除轴标签，但保留刻度
ax_left_fE.set_xlabel('')
ax_left_fE_twin.set_xlabel('')

# 下方条带小图（经度）
ax_bottom_fE = divider_fE.append_axes("bottom", size="20%", pad=0.1, sharex=axes[0])
ax_bottom_fE_twin = ax_bottom_fE.twinx()

# 绘制数据量柱状图
ax_bottom_fE.bar(lon_group_fE['lon_center'], lon_group_fE['data_amount'], width=2.0, color='gray', alpha=0.7)

# 绘制平均值折线图
ax_bottom_fE_twin.plot(lon_group_fE['lon_center'], lon_group_fE['f_E_avg_smooth'], color='red', linewidth=1.25)

# 移除轴标签，但保留刻度
ax_bottom_fE.set_ylabel('')
ax_bottom_fE_twin.set_ylabel('')

# 对于 f_NE 图
divider_fNE = make_axes_locatable(axes[1])

# 左侧条带小图（纬度）
ax_left_fNE = divider_fNE.append_axes("left", size="20%", pad=0.1, sharey=axes[1])
ax_left_fNE_twin = ax_left_fNE.twiny()

# 绘制数据量柱状图
ax_left_fNE.barh(lat_group_fE['lat_center'], lat_group_fE['data_amount'], height=2.0, color='gray', alpha=0.7)

# 绘制平均值折线图
ax_left_fNE_twin.plot(lat_group_fE['f_NE_avg_smooth'], lat_group_fE['lat_center'], color='green', linewidth=1.25)

# 移除轴标签，但保留刻度
ax_left_fNE.set_xlabel('')
ax_left_fNE_twin.set_xlabel('')

# 下方条带小图（经度）
ax_bottom_fNE = divider_fNE.append_axes("bottom", size="20%", pad=0.1, sharex=axes[1])
ax_bottom_fNE_twin = ax_bottom_fNE.twinx()

# 绘制数据量柱状图
ax_bottom_fNE.bar(lon_group_fE['lon_center'], lon_group_fE['data_amount'], width=2.0, color='gray', alpha=0.7)

# 绘制平均值折线图
ax_bottom_fNE_twin.plot(lon_group_fE['lon_center'], lon_group_fE['f_NE_avg_smooth'], color='green', linewidth=1.25)

# 移除轴标签，但保留刻度
ax_bottom_fNE.set_ylabel('')
ax_bottom_fNE_twin.set_ylabel('')

# 调整刻度和轴范围
for ax in [ax_left_fE, ax_left_fE_twin, ax_left_fNE, ax_left_fNE_twin]:
    ax.grid(False)
    ax.tick_params(axis='both', which='both', length=5, width=0.75)  # 修改线宽为0.75
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)  # 修改脊线宽度为0.75
        spine.set_color('black')

for ax in [ax_bottom_fE, ax_bottom_fE_twin, ax_bottom_fNE, ax_bottom_fNE_twin]:
    ax.grid(False)
    ax.tick_params(axis='both', which='both', length=5, width=0.75)  # 修改线宽为0.75
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)  # 修改脊线宽度为0.75
        spine.set_color('black')

plt.setp(ax_left_fE.get_yticklabels(), visible=False)
plt.setp(ax_left_fNE.get_yticklabels(), visible=False)
plt.setp(ax_bottom_fE.get_xticklabels(), visible=False)
plt.setp(ax_bottom_fNE.get_xticklabels(), visible=False)

# 修正右侧 f_NE 图的右侧标签位置
axes[1].tick_params(axis='y', which='both', labelright=True, right=True)
axes[1].yaxis.set_label_position("right")

# 确保所有子图的布局不重叠
plt.tight_layout(rect=[0, 0.1, 1, 1])  # 留出底部空间给 colorbar

# 保存图像为高分辨率PNG文件
plt.savefig('China_fE_fNE_Distribution_with_Strips_Adjusted.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()
