#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 CSV 文件转换为 dataset_exe_result.json 格式的脚本
"""

import csv
import json
import os


def convert_csv_to_json(csv_path, sql_id_prefix="sql", sql_id_number=1):
    """
    将 CSV 文件转换为 dataset_exe_result.json 格式并在控制台打印
    
    Args:
        csv_path: CSV 文件路径
        sql_id_prefix: SQL ID 前缀，默认为 "sql"
        sql_id_number: SQL ID 编号，默认为 1
    """
    # 读取 CSV 文件
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except FileNotFoundError:
        print(f"错误: CSV 文件不存在: {csv_path}")
        return
    except Exception as e:
        print(f"错误: 读取 CSV 文件时出错: {e}")
        return
    
    if not rows:
        print("CSV 文件为空")
        return
    
    # 获取列名
    columns = list(rows[0].keys())
    
    # 生成 SQL 查询语句
    column_list = ', '.join(columns)
    sql_query = f"SELECT {column_list}\nFROM user_activity_stats\nORDER BY {columns[0]}, {columns[1]};"
    
    # 转换数据格式
    result = []
    for row in rows:
        # 将每行数据转换为字典，保持原始值
        result_row = {}
        for key, value in row.items():
            result_row[key] = value
        result.append(result_row)
    
    # 生成 sql_id
    sql_id = f"{sql_id_prefix}_{sql_id_number}"
    
    # 构建新的条目
    new_entry = {
        "sql_id": sql_id,
        "sql": sql_query,
        "result": result
    }
    
    # 在控制台打印格式化后的 JSON
    json_output = json.dumps([new_entry], ensure_ascii=False, indent=4)
    print(json_output)


def main():
    """主函数"""
    # 输入文件路径
    csv_path = r"C:\Users\11386\Downloads\data.csv"
    
    # 检查 CSV 文件是否存在
    if not os.path.exists(csv_path):
        print(f"错误: CSV 文件不存在: {csv_path}")
        return
    
    # 执行转换并打印 JSON
    convert_csv_to_json(csv_path)


if __name__ == "__main__":
    main()

