import os
import requests
import logging
from tqdm import tqdm
import datetime

# 配置日志记录
logging.basicConfig(filename='mod08_download.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger()

API_TOKEN_FILE = 'my_NASA_API_token'
MOD08_DOWNLOAD_LIST_FILE = 'MOD08_download_list.txt'


def load_nasa_api_token(token_file=API_TOKEN_FILE):
    with open(token_file, 'r') as file:
        token = file.read().strip()
        return token


API_TOKEN = load_nasa_api_token()

session = requests.Session()
session.headers.update({'Authorization': f'Bearer {API_TOKEN}'})


def find_closest_file(urls, target_date):
    target_str = target_date.strftime("%Y%j")
    for url in urls:
        file_name = url.split('/')[-1]
        date_str = file_name.split('.')[1][1:]
        if date_str == target_str:
            return url, file_name
    return None, None


def download_file(url, filename, download_dir):
    filename = filename.strip()
    local_filepath = os.path.join(download_dir, filename)
    print("加载API中")

    response = session.get(url.strip(), stream=True)
    response.raise_for_status()
    print("加载成功")

    total_size = int(response.headers.get('content-length', 0))
    with open(local_filepath, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            pbar.update(len(data))

    return local_filepath


def download_mod08_data(date, download_dir):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    with open(MOD08_DOWNLOAD_LIST_FILE, 'r') as f:
        urls = f.readlines()

    url, file_name = find_closest_file(urls, date)
    print("配置url:", url)

    if url and file_name:
        new_filename = f"MOD08_{date.strftime('%Y%m%d')}.hdf"
        new_filepath = os.path.join(download_dir, new_filename)
        temp_filepath = os.path.join(download_dir, file_name)

        # 检查是否已存在完整文件
        if os.path.exists(new_filepath):
            print(f"文件 {new_filename} 已存在，无需重新下载。")
            return new_filepath
        # 检查是否有下载失败的文件
        elif os.path.exists(temp_filepath):
            print(f"发现未完成的文件 {file_name}，将其删除并重新下载。")
            os.remove(temp_filepath)

        # 进行文件下载
        downloaded_file_path = download_file(url.strip(), file_name, download_dir)
        if downloaded_file_path:
            os.rename(downloaded_file_path, new_filepath)
            return new_filepath
    return None


if __name__ == '__main__':
    start_date = datetime.date(2015, 7, 7)
    end_date = datetime.date(2021, 12, 31)

    current_date = start_date
    while current_date <= end_date:
        print(f"Downloading data for {current_date}")

        download_dir = "downloaded_data\\mod08"
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        download_mod08_data(current_date, download_dir)
        current_date += datetime.timedelta(days=1)
