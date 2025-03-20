def parse_query_data(file_path):
    """解析文件，生成每个查询的整数列表"""
    with open(file_path, 'r') as f:
        while True:
            # 读取标题行（例如 "Reduced ids for Query 0"）
            header = f.readline()
            if not header:
                break  # 文件结束
            
            if not header.startswith("Reduced ids for Query"):
                raise ValueError(f"文件格式错误：无效的标题行 '{header.strip()}'")
            
            # 提取查询编号
            try:
                query_num = int(header.split()[-1])
            except (IndexError, ValueError):
                raise ValueError(f"无法解析查询编号：'{header.strip()}'")
            
            # 读取数据行并解析整数列表
            data_line = f.readline().strip()
            try:
                data = list(map(int, data_line.split()))
            except ValueError:
                raise ValueError(f"数据格式错误：'{data_line}'")
            
            yield query_num, data

def compare_files(file1, file2):
    """比较两个文件，返回差异信息"""
    try:
        # 解析两个文件
        data1 = list(parse_query_data(file1))
        data2 = list(parse_query_data(file2))
    except ValueError as e:
        print(f"错误：{e}")
        return False
    
    # 检查查询数量
    if len(data1) != 16 or len(data2) != 16:
        print(f"错误：查询数量不正确（应为16，实际为 {len(data1)} 和 {len(data2)}）")
        return False
    
    differences = []
    
    # 逐个查询比较
    for (q1_num, q1_data), (q2_num, q2_data) in zip(data1, data2):
        # 检查查询编号是否一致
        if q1_num != q2_num:
            differences.append(f"查询编号不匹配：文件1={q1_num}，文件2={q2_num}")
            continue
        
        # 检查数据长度
        if len(q1_data) != 8000 or len(q2_data) != 8000:
            differences.append(f"查询 {q1_num} 数据数量不正确（文件1={len(q1_data)}，文件2={len(q2_data)}）")
            continue
        
        # 逐个元素比较
        for idx in range(8000):
            val1 = q1_data[idx]
            val2 = q2_data[idx]
            if val1 != val2:
                differences.append(f"查询 {q1_num} 索引 {idx}：文件1={val1}，文件2={val2}")
    
    return differences

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("用法：python compare_queries.py 文件1 文件2")
        sys.exit(1)
    
    file1, file2 = sys.argv[1], sys.argv[2]
    
    differences = compare_files(file1, file2)
    
    if not differences:
        print("文件内容完全相同。")
    else:
        print("发现以下差异：")
        for diff in differences:
            print(f"  - {diff}")
        print(f"总计 {len(differences)} 处差异。")