import json
import os
import sys
from toon_format import encode
from tqdm import tqdm




def _store_toon_schema_to_chromaDB(encoded_schemas, merged_data, chroma_path, workspace_root):
    """将TOON编码的表结构存储到ChromaDB"""
    try:
        import chromadb
    except ImportError:
        print("错误：需要安装chromadb库。请在运行此脚本前安装。")
        sys.exit(1)
    
    try:
        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_or_create_collection(name="schema_knowledge")
        print(f"准备存储TOON编码的表结构数据...")

        # 准备documents、metadatas和ids
        documents = []
        metadatas = []
        ids = []
        
        # 为每个表存储TOON编码
        for table_name, encoded_schema in tqdm(encoded_schemas.items(), desc="存储表结构", unit="表"):
            table_info = merged_data.get(table_name, {})
            fields = table_info.get("fields", {})
            
            # 存储TOON编码
            documents.append(encoded_schema)
            ids.append(f"table_{table_name}")
            metadatas.append({
                "type": "table",
                "table_name": table_name,
                "source": "merged_schema_analysis",
                "encoding_format": "TOON",
                "field_count": str(len(fields)),
            })
        
        print(f"生成了 {len(documents)} 个表结构文档")
        
        # 存储开发者查看的载荷示例
        temp_payload = {
            "documents": documents[:5],  # 只保存前5个作为示例
            "metadatas": metadatas[:5],
            "ids": ids[:5],
            "total_count": len(documents),
        }
        temp_output_path = os.path.join(workspace_root, "T3", "temp", "schema_payload_sample.json")
        with open(temp_output_path, 'w', encoding='utf-8') as f:
            json.dump(temp_payload, f, ensure_ascii=False, indent=2)
        print(f"Schema载荷示例已保存到: {temp_output_path}")

        # 清理旧数据
        try:
            existing_count = collection.count()
            if existing_count > 0:
                print(f"清理 {existing_count} 个旧文档...")
                existing_data = collection.get()
                if existing_data and existing_data.get("ids"):
                    collection.delete(ids=existing_data["ids"])
        except Exception as cleanup_exc:
            print(f"警告: 清理旧数据时出错（不影响新数据存储）: {cleanup_exc}")
        
        # 存储新数据
        collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
        print(f"TOON编码的表结构数据存储完成，共存储 {len(documents)} 个表。")
        
    except Exception as exc:
        print(f"无法将Schema数据存储到ChromaDB: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)




def merge_schema_data(mschema_data, schema_data):
    """合并两个schema数据源"""
    merged_data = {}
    
    # 从 mschema_database_main.json 获取详细的表结构
    if "tables" in mschema_data:
        for table_name, table_info in mschema_data["tables"].items():
            merged_data[table_name] = {
                "table_name": table_name,
                "fields": table_info.get("fields", {}),
                "comment": table_info.get("comment", ""),
                "examples": table_info.get("examples", [])
            }
    
    # 从 schema.json 获取表描述信息
    schema_lookup = {item["table_name"]: item for item in schema_data}
    
    # 合并描述信息
    for table_name, table_info in merged_data.items():
        if table_name in schema_lookup:
            schema_info = schema_lookup[table_name]
            table_info["table_description"] = schema_info.get("table_description", "")
            table_info["columns"] = schema_info.get("columns", [])
            
            # 将columns中的description信息合并到fields的comment字段中
            columns = schema_info.get("columns", [])
            fields = table_info.get("fields", {})
            
            # 创建字段名到description的映射
            column_desc_map = {col["col"]: col.get("description", "") for col in columns}
            
            # 为每个字段添加description到comment字段
            for field_name, field_info in fields.items():
                if field_name in column_desc_map:
                    description = column_desc_map[field_name]
                    # 如果字段已有comment，添加description信息
                    existing_comment = field_info.get("comment", "")
                    if existing_comment and description:
                        field_info["comment"] = f"{existing_comment} (字段描述: {description})"
                    elif description:
                        field_info["comment"] = description
                    # 如果没有description但有existing_comment，保持不变
                    # 如果都没有，保持字段原有的comment（如果有的话）
    
    return merged_data


def analyze_schema():
    """分析schema数据并生成编码结果"""
    # 获取路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    # 文件路径
    mschema_path = os.path.join(workspace_root, "T3", "data", "mschema_database_main.json")
    schema_desc_path = os.path.join(workspace_root, "T3", "data", "schema.json")
    
    # 输出目录
    output_dir = os.path.join(workspace_root, "T3", "temp")
    os.makedirs(output_dir, exist_ok=True)
    
    # ChromaDB配置
    chroma_path = os.path.join(workspace_root, "T3", "chroma_db")
    
    # 加载数据文件
    print("正在加载 mschema_database_main.json...")
    with open(mschema_path, 'r', encoding='utf-8') as f:
        mschema_data = json.load(f)
    
    print("正在加载 schema.json...")
    with open(schema_desc_path, 'r', encoding='utf-8') as f:
        schema_data = json.load(f)

    # 合并数据
    print("正在合并schema数据...")
    merged_data = merge_schema_data(mschema_data, schema_data)
    
    # 按表分别进行TOON编码
    print("正在对各个表进行TOON编码...")
    encoded_schemas = {}
    
    for table_name, table_info in tqdm(merged_data.items(), desc="TOON编码", unit="表"):
        try:
            # 为每个表创建独立的编码数据
            table_data = {
                "table_name": table_name,
                "table_description": table_info.get("table_description", ""),
                "comment": table_info.get("comment", ""),
                "fields": table_info.get("fields", {}),
                "columns": table_info.get("columns", [])
            }
            
            encoded_schema = encode(table_data)
            encoded_schemas[table_name] = encoded_schema
            
            # 清理TOON编码结果，去掉重复的columns部分
            lines = encoded_schema.split('\n')
            cleaned_lines = []
            skip_columns = False
            
            for line in lines:
                if line.strip().startswith('columns['):
                    skip_columns = True
                    continue
                if skip_columns and line.strip() == '':
                    skip_columns = False
                    continue
                if not skip_columns:
                    cleaned_lines.append(line)
            
            cleaned_toon = '\n'.join(cleaned_lines)
            
            # 保存每个表的TOON编码结果
            toon_output_path = os.path.join(output_dir, f"encoded_schema_table_{table_name}.toon")
            with open(toon_output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_toon)
            
        except Exception as e:
            print(f"    编码表 {table_name} 时出错: {e}")
            continue
    
    # 存储TOON编码的schema到ChromaDB
    _store_toon_schema_to_chromaDB(encoded_schemas, merged_data, chroma_path, workspace_root)
    
    # 保存开发者查看文件（包含统计信息）
    final_output_path = os.path.join(output_dir, "final_schema_with_toon.json")
    with open(final_output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "encoded_tables_count": len(encoded_schemas),
            "encoding_format": "TOON",
            "storage_status": "inserted_to_chromaDB",
            "tables": list(encoded_schemas.keys()),
            "merged_data_summary": {
                table_name: {
                    "field_count": len(info.get("fields", {})),
                    "description": info.get("table_description", "")
                }
                for table_name, info in merged_data.items()
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"开发者查看文件已保存到: {final_output_path}")


if __name__ == "__main__":
    print("开始分析Schema数据...")
    analyze_schema()
    print("Schema分析完成！")
