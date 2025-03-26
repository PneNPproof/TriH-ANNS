import hashlib
import sys

def calculate_file_hash(file_path, hash_algorithm="sha256", chunk_size=4096):
    """
    计算文件的哈希值。
    
    :param file_path: 文件路径
    :param hash_algorithm: 哈希算法（默认为 'sha256'）
    :param chunk_size: 每次读取的块大小（默认 4KB）
    :return: 文件的哈希值（十六进制字符串）
    """
    # 创建哈希对象
    hash_func = hashlib.new(hash_algorithm)
    
    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):  # 按块读取文件
                hash_func.update(chunk)
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 不存在！")
        sys.exit(1)
    except IOError:
        print(f"错误：无法读取文件 '{file_path}'！")
        sys.exit(1)
    
    return hash_func.hexdigest()

def compare_files(file1, file2, hash_algorithm="sha256"):
    """
    比较两个文件的哈希值。
    
    :param file1: 第一个文件路径
    :param file2: 第二个文件路径
    :param hash_algorithm: 哈希算法（默认为 'sha256'）
    :return: 如果文件相同返回 True，否则返回 False
    """
    hash1 = calculate_file_hash(file1, hash_algorithm)
    hash2 = calculate_file_hash(file2, hash_algorithm)

    print(f"文件 1 的 {hash_algorithm.upper()} 哈希值: {hash1}")
    print(f"文件 2 的 {hash_algorithm.upper()} 哈希值: {hash2}")

    if hash1 == hash2:
        print("两个文件的内容相同。")
        return True
    else:
        print("两个文件的内容不同。")
        return False

if __name__ == "__main__":
    # 检查命令行参数数量
    if len(sys.argv) < 3:
        print("用法：python compare_hashes.py 文件1 文件2 [哈希算法]")
        print("示例：python compare_hashes.py file1.txt file2.txt sha256")
        sys.exit(1)

    # 获取命令行参数
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    algorithm = sys.argv[3].lower() if len(sys.argv) > 3 else "sha256"

    # 检查是否支持该哈希算法
    if algorithm not in hashlib.algorithms_available:
        print(f"错误：不支持的哈希算法 '{algorithm}'！")
        print(f"可用的哈希算法有：{', '.join(sorted(hashlib.algorithms_available))}")
        sys.exit(1)

    # 比较文件
    compare_files(file1, file2, algorithm)