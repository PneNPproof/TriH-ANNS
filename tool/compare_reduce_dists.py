def parse_float_query_data(file_path):
    """解析浮点数查询数据，返回生成器：(query_num, [float_list])"""
    with open(file_path, 'r') as f:
        while True:
            # 读取标题行
            header = f.readline()
            if not header:
                break
            
            # 验证标题格式
            if not header.startswith("Reduced dists for Query "):
                raise ValueError(f"无效的标题行: {header.strip()}")
            
            # 解析查询编号
            try:
                query_num = int(header.split()[-1])
            except (ValueError, IndexError):
                raise ValueError(f"无法解析查询编号: {header.strip()}")
            
            # 读取数据行
            data_line = f.readline().strip()
            try:
                # 解析浮点数并保留四位小数
                data = [round(float(x), 4) for x in data_line.split()]
            except ValueError:
                raise ValueError(f"数据格式错误: {data_line}")
            
            # 验证数据数量
            if len(data) != 8000:
                raise ValueError(f"查询 {query_num} 数据数量错误: {len(data)}")
            
            yield query_num, data

def compare_float_files(file1, file2):
    """比较两个浮点数文件，返回差异列表"""
    try:
        data1 = list(parse_float_query_data(file1))
        data2 = list(parse_float_query_data(file2))
    except ValueError as e:
        return f"文件解析错误: {str(e)}"
    
    # 验证查询数量
    if len(data1) != 16 or len(data2) != 16:
        return f"查询数量错误: 文件1有{len(data1)}个，文件2有{len(data2)}个"
    
    differences = []
    
    for (q1_num, q1_data), (q2_num, q2_data) in zip(data1, data2):
        # 检查查询编号一致性
        if q1_num != q2_num:
            differences.append(f"查询编号不匹配: 文件1={q1_num} vs 文件2={q2_num}")
            continue
        
        # 逐个元素比较
        for idx in range(8000):
            val1 = q1_data[idx]
            val2 = q2_data[idx]
            
            if val1 != val2:
                differences.append(
                    f"查询 {q1_num} 索引 {idx}: "
                    f"文件1={val1:.4f} vs 文件2={val2:.4f}"
                )
    
    return differences

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("用法: python compare_floats.py 文件1 文件2")
        sys.exit(1)
    
    file1, file2 = sys.argv[1], sys.argv[2]
    
    try:
        diffs = compare_float_files(file1, file2)
    except Exception as e:
        print(f"比较失败: {str(e)}")
        sys.exit(1)
    
    if not diffs:
        print("文件内容完全一致")
    else:
        print(f"发现 {len(diffs)} 处差异:")
        for diff in diffs:
            print(f"  - {diff}")