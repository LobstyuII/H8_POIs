import os
import ftplib
import logging
import datetime
import time
import pandas as pd
import numpy as np
import netCDF4 as nc
from concurrent.futures import ThreadPoolExecutor, as_completed

# 设置日志配置
logging.basicConfig(filename='h8_download.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# FTP连接信息
FTP_ADDRESS = "ftp.ptree.jaxa.jp"
FTP_UID = "511153727_qq.com"
FTP_PW = "SP+wari8"

# 下载文件的函数
def download_from_ftp(ftp_path, local_filename, download_dir, max_retries=5, retry_delay=10):
    logger.info(f"开始从FTP服务器下载文件: {ftp_path}")

    # 创建下载目录
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    local_filepath = os.path.join(download_dir, local_filename)
    resume_byte_pos = 0

    # 如果文件已经存在，检查文件大小用于断点续传
    if os.path.exists(local_filepath):
        resume_byte_pos = os.path.getsize(local_filepath)
        logger.info(f"文件已存在，准备从 {resume_byte_pos} 字节处继续下载。")

    attempt = 0
    while attempt < max_retries:
        try:
            with ftplib.FTP(FTP_ADDRESS, timeout=retry_delay) as ftp:
                ftp.login(FTP_UID, FTP_PW)
                ftp.voidcmd("TYPE I")
                file_size = ftp.size(ftp_path)

                # 检查文件是否已经下载完成
                if resume_byte_pos >= file_size:
                    logger.info("文件已完整下载，无需重新下载。")
                    return local_filepath

                # 打开本地文件，以追加模式（'ab'）写入
                with open(local_filepath, 'ab') as local_file:
                    # 使用断点续传继续下载
                    def callback(data):
                        local_file.write(data)

                    # 重新开始下载并使用断点续传
                    ftp.retrbinary(f'RETR {ftp_path}', callback, rest=resume_byte_pos)

                # 检查文件大小是否正确
                if os.path.getsize(local_filepath) == file_size:
                    logger.info(f"文件下载成功: {local_filepath}")
                    return local_filepath
                else:
                    raise Exception("下载文件大小不匹配")

        except ftplib.all_errors as e:
            logger.error(f"FTP文件下载失败: {e}, 重试 {attempt + 1}/{max_retries}...")
            attempt += 1
            time.sleep(retry_delay)

    logger.error(f"文件下载失败: 超过最大重试次数 {max_retries}")
    return None

# 处理L2ARP文件的函数
def process_l2arp_file(l2arp_file_path, lookup_df, output_file_path):
    try:
        # 打开L2ARP文件
        dataset = nc.Dataset(l2arp_file_path, 'r')

        # 获取QA_flag变量
        qa_flag = dataset.variables['QA_flag'][:]

        # 创建用于存储结果的列表
        output_rows = []

        # 遍历每个站点，提取数据
        for _, row in lookup_df.iterrows():
            station_name = row['Station']
            l2arp_x = int(row['L2ARP_x'])
            l2arp_y = int(row['L2ARP_y'])

            # 获取对应站点的QA_flag值
            qa_value = qa_flag[l2arp_y, l2arp_x]

            # 提取前三个比特位的值
            data_availability = (qa_value & 1)  # 第0位
            land_water_flag = (qa_value >> 1) & 1  # 第1位
            cloud_flag = (qa_value >> 2) & 1  # 第2位

            # 将数据添加到结果列表中
            output_rows.append({
                'Station': station_name,
                'Data Availability': data_availability,
                'Land/Water Flag': land_water_flag,
                'Cloud Flag': cloud_flag
            })

        # 将结果保存为DataFrame
        output_df = pd.DataFrame(output_rows)

        # 将结果保存为CSV文件
        output_df.to_csv(output_file_path, index=False)

        # 关闭L2ARP文件
        dataset.close()

        return True  # 处理成功
    except Exception as e:
        logger.error(f"处理文件时发生错误: {e}")
        return False  # 处理失败

# 下载并处理文件的任务
def download_and_process(date, hour, minute, lookup_df, download_dir, output_base_dir):
    # 设置文件名和路径
    ftp_path = f"/pub/himawari/L2/ARP/030/{date.year:04d}{date.month:02d}/{date.day:02d}/{hour:02d}/NC_H08_{date.year:04d}{date.month:02d}{date.day:02d}_{hour:02d}{minute:02d}_L2ARP030_FLDK.02401_02401.nc"
    local_filename = f"himawari_{date.strftime('%Y%m%d')}_{hour:02d}{minute:02d}.nc"
    local_filepath = os.path.join(download_dir, local_filename)

    # 创建以日期为名称的输出目录
    daily_output_dir = os.path.join(output_base_dir, date.strftime('%Y%m%d'))
    os.makedirs(daily_output_dir, exist_ok=True)
    output_file_path = os.path.join(daily_output_dir, f"poi_h8l2arp_{date.strftime('%Y%m%d')}_{hour:02d}{minute:02d}.csv")

    # 检查是否已经存在处理完成的文件
    if os.path.exists(output_file_path):
        logger.info(f"处理后的文件已存在，跳过: {output_file_path}")
        print(f"{datetime.datetime.now()} - 跳过文件（已存在）：{output_file_path}")
        return

    # 下载文件
    downloaded_file = download_from_ftp(ftp_path, local_filename, download_dir)

    # 如果文件下载成功，则处理文件
    if downloaded_file:
        if process_l2arp_file(downloaded_file, lookup_df, output_file_path):
            logger.info(f"文件处理完成，结果保存在: {output_file_path}")
            print(f"{datetime.datetime.now()} - 文件处理完成：{output_file_path}")

            # 删除下载的原始数据文件
            os.remove(downloaded_file)
            logger.info(f"已删除下载的原始数据文件: {downloaded_file}")
        else:
            # 处理失败，重新下载文件
            os.remove(downloaded_file)  # 删除下载的文件
            print(f"{datetime.datetime.now()} - 处理文件失败，重新下载：{ftp_path}")
            download_and_process(date, hour, minute, lookup_df, download_dir, output_base_dir)
    else:
        logger.error(f"无法下载文件: {ftp_path}")
        print(f"{datetime.datetime.now()} - 无法下载文件：{ftp_path}")

# 主函数
def main():
    lookup_df = pd.read_csv('station_xy_lookup_table.csv')
    download_dir = "downloaded_data/h8l2arp"
    output_base_dir = "output/h8l2arp"
    os.makedirs(download_dir, exist_ok=True)

    # 设置日期范围和时间
    start_date = datetime.date(2015, 7, 15)
    end_date = datetime.date(2021, 12, 31)
    hours = list(range(1, 9))  # 01:00 - 08:00 UTC
    minutes = [0, 10, 20, 30, 40, 50]

    # 准备任务列表
    tasks = []
    current_date = start_date
    while current_date <= end_date:
        for hour in hours:
            for minute in minutes:
                tasks.append((current_date, hour, minute))
        current_date += datetime.timedelta(days=1)

    # 使用线程池并行下载和处理
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(download_and_process, date, hour, minute, lookup_df, download_dir, output_base_dir)
                   for (date, hour, minute) in tasks]

        # 等待所有任务完成
        for future in as_completed(futures):
            future.result()  # 捕获异常

if __name__ == '__main__':
    main()
