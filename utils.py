import zipfile
import os

def unzip_file(zip_path, extract_to):
    # 确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    # 打开 zip 文件并解压
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"文件已解压到：{extract_to}")

# 示例：解压文件
zip_file_path = 'transformers-4.51.3-Qwen2.5-Omni-preview.zip'  # 这里替换为你自己的 zip 文件路径
output_directory = '.'
unzip_file(zip_file_path, output_directory)
