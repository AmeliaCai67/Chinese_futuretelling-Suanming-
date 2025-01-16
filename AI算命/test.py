import os
import requests
from tqdm import tqdm
import time

def download_with_retry(url, output_path, max_retries=5):
    """带重试机制的下载函数"""
    for attempt in range(max_retries):
        try:
            # 设置较长的超时时间
            response = requests.get(url, stream=True, timeout=30)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            return True
        except Exception as e:
            print(f"尝试 {attempt + 1}/{max_retries} 失败: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 指数退避
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
    return False

def download_model_files():
    # 设置代理（如果需要）
    proxies = {
        'http': 'http://127.0.0.1:7890',
        'https': 'http://127.0.0.1:7890'
    }
    
    # 模型文件及其镜像URL（提供多个备选地址）
    files = {
        "pytorch_model.bin": [
            "https://huggingface.co/shibing624/text2vec-base-chinese/resolve/main/pytorch_model.bin",
            "https://mirror.xyz/models/shibing624/text2vec-base-chinese/pytorch_model.bin"  # 示例镜像
        ],
        "config.json": [
            "https://huggingface.co/shibing624/text2vec-base-chinese/resolve/main/config.json"
        ],
        "vocab.txt": [
            "https://huggingface.co/shibing624/text2vec-base-chinese/resolve/main/vocab.txt"
        ],
        "special_tokens_map.json": [
            "https://huggingface.co/shibing624/text2vec-base-chinese/resolve/main/special_tokens_map.json"
        ],
        "tokenizer_config.json": [
            "https://huggingface.co/shibing624/text2vec-base-chinese/resolve/main/tokenizer_config.json"
        ]
    }
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "embedding", "text2vec-base-chinese")
    os.makedirs(model_dir, exist_ok=True)
    
    for filename, urls in files.items():
        output_path = os.path.join(model_dir, filename)
        print(f"\n开始下载 {filename}...")
        
        success = False
        for url in urls:
            print(f"尝试从 {url} 下载...")
            if download_with_retry(url, output_path):
                success = True
                break
        
        if not success:
            print(f"下载 {filename} 失败！")
            return False
    
    return True

if __name__ == "__main__":
    print("开始下载模型文件...")
    
    # 如果需要设置代理环境变量
    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
    
    if download_model_files():
        print("所有文件下载成功！")
    else:
        print("下载失败，请检查网络连接或尝试使用代理。")