import os
import zipfile
import requests
import subprocess
from urllib.parse import urlparse, unquote
import shutil


def list_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            print(os.path.join(root, file))

# 获取环境变量中的密码
password = os.getenv('ZIP_PASSWORD')

def download_and_setup():
    file_path = "Gomoku.zip"

      # 解压文件
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        # 解压到当前目录
        extract_dir = os.path.dirname(file_path)
        zip_ref.extractall(extract_dir,pwd=bytes(password, 'utf-8'))
        #extracted_files_and_dirs = os.listdir(extract_dir)

    # 获取当前目录
    current_dir = os.path.dirname(file_path)
    print(current_dir)
    src_dir = os.path.join(current_dir, 'Guess_the_riddles')
    #dst_dir = os.path.join(current_dir, '..')
    print(src_dir)
    #print(dst_dir)

    # 遍历解压后的文件夹中的所有文件和目录
    for item in os.listdir(src_dir):
        # 构造完整的文件或目录路径
        src_item = os.path.join(src_dir, item)
        dst_item = os.path.join(current_dir, item)

        # 如果目标路径存在，删除它
        if os.path.exists(dst_item):
            if os.path.isfile(dst_item):
                os.remove(dst_item)
            else:
                shutil.rmtree(dst_item)

        # 移动文件或目录
        shutil.move(src_item, dst_item)

    print("zip file extracted")

    list_files(os.getcwd())


    # 设置PYTHONPATH环境变量
    pythonpath = os.path.dirname(file_path)
    os.environ['PYTHONPATH'] = pythonpath
    print(pythonpath)
    
    # 安装requirements.txt中的依赖
    requirements_path = os.path.join(pythonpath, 'requirements.txt')
    print(requirements_path)
    if os.path.exists(requirements_path):
        subprocess.run(['pip', 'install', '-r', requirements_path], check=True)

    # 运行appBot.py
    app_bot_path = os.path.join(pythonpath, 'app.py')
    print(app_bot_path)
    if os.path.exists(app_bot_path):
        subprocess.run(['python', app_bot_path], check=True)
    else:
        raise Exception(f"app.py does not exist in {pythonpath}")


# 使用示例
download_and_setup()

