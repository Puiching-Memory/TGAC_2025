"""
将SQL数据库中的指定表导出为CSV文件，支持导出字段注释。

使用方法:
    python export_table_to_csv.py <table_name> [选项]

示例:
    # 使用默认配置导出表（包含字段注释）
    python export_table_to_csv.py dim_extract_311381_conf
    
    # 指定输出文件
    python export_table_to_csv.py dws_argothek_ce1_login_di --output output.csv
    
    # 指定数据库连接信息
    python export_table_to_csv.py my_table --host 127.0.0.1 --port 9030 --user root --database database_main
    
    # 限制导出行数（用于测试）
    python export_table_to_csv.py large_table --limit 1000
    
    # 不导出字段注释
    python export_table_to_csv.py my_table --no-comments

注意:
    - 默认情况下，CSV文件会在列名行下方添加一行字段注释
    - 如果字段没有注释，注释行中对应位置为空
    - 使用 --no-comments 参数可以禁用注释导出
"""
import csv
import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Any
from decimal import Decimal
from datetime import datetime, date

try:
    import pymysql
except ImportError:
    print("错误: 需要安装 pymysql 库")
    print("请运行: pip install pymysql")
    sys.exit(1)


class TableExporter:
    """数据库表导出器"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 9030, 
                 user: str = "root", password: str = "", 
                 database: str = "database_main"):
        """
        初始化数据库连接配置
        
        Args:
            host: 数据库主机地址
            port: 数据库端口
            user: 数据库用户名
            password: 数据库密码
            database: 数据库名称
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
    
    def connect(self) -> None:
        """建立数据库连接"""
        try:
            self.connection = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            print(f"✓ 成功连接到数据库: {self.database}@{self.host}:{self.port}")
        except Exception as e:
            print(f"✗ 连接数据库失败: {e}")
            sys.exit(1)
    
    def close(self) -> None:
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            print("✓ 数据库连接已关闭")
    
    def table_exists(self, table_name: str) -> bool:
        """检查表是否存在"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) as count FROM information_schema.tables "
                    "WHERE table_schema = %s AND table_name = %s",
                    (self.database, table_name)
                )
                result = cursor.fetchone()
                return result['count'] > 0
        except Exception as e:
            print(f"✗ 检查表是否存在时出错: {e}")
            return False
    
    def get_table_row_count(self, table_name: str) -> int:
        """获取表的行数"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) as count FROM `{table_name}`")
                result = cursor.fetchone()
                return result['count']
        except Exception as e:
            print(f"✗ 获取表行数时出错: {e}")
            return 0
    
    def get_table_columns(self, table_name: str) -> list:
        """获取表的列名"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(f"DESCRIBE `{table_name}`")
                columns = cursor.fetchall()
                return [col['Field'] for col in columns]
        except Exception as e:
            print(f"✗ 获取表列信息时出错: {e}")
            return []
    
    def get_column_comments(self, table_name: str) -> Dict[str, str]:
        """
        获取表的字段注释
        
        Args:
            table_name: 表名
            
        Returns:
            字段名到注释的字典映射
        """
        comments = {}
        try:
            with self.connection.cursor() as cursor:
                # 查询information_schema获取字段注释
                query = """
                    SELECT COLUMN_NAME, COLUMN_COMMENT
                    FROM information_schema.COLUMNS
                    WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
                    ORDER BY ORDINAL_POSITION
                """
                cursor.execute(query, (self.database, table_name))
                results = cursor.fetchall()
                for row in results:
                    column_name = row['COLUMN_NAME']
                    comment = row['COLUMN_COMMENT'] or ""
                    comments[column_name] = comment
        except Exception as e:
            # 如果查询失败，尝试使用SHOW FULL COLUMNS
            try:
                with self.connection.cursor() as cursor:
                    cursor.execute(f"SHOW FULL COLUMNS FROM `{table_name}`")
                    results = cursor.fetchall()
                    for row in results:
                        column_name = row['Field']
                        comment = row.get('Comment', '') or ""
                        comments[column_name] = comment
            except Exception as e2:
                print(f"⚠ 获取字段注释时出错: {e2}，将不包含注释信息")
        return comments
    
    def convert_value(self, value: Any) -> str:
        """
        转换数据库值为CSV兼容的字符串
        
        Args:
            value: 数据库值
            
        Returns:
            转换后的字符串
        """
        if value is None:
            return ""
        elif isinstance(value, (Decimal, float)):
            # 处理Decimal和float，避免科学计数法
            if isinstance(value, Decimal):
                value = float(value)
            # 如果是整数，返回整数格式
            if value == int(value):
                return str(int(value))
            return str(value)
        elif isinstance(value, (datetime, date)):
            # 日期时间转为ISO格式
            return value.isoformat()
        elif isinstance(value, bytes):
            # 二进制数据转为十六进制字符串
            return value.hex()
        else:
            return str(value)
    
    def export_table(self, table_name: str, output_file: str, 
                    limit: Optional[int] = None, batch_size: int = 10000,
                    include_comments: bool = True) -> None:
        """
        导出表数据到CSV文件
        
        Args:
            table_name: 表名
            output_file: 输出CSV文件路径
            limit: 限制导出的行数（None表示导出全部）
            batch_size: 批量查询的大小
        """
        # 检查表是否存在
        if not self.table_exists(table_name):
            print(f"✗ 错误: 表 '{table_name}' 不存在于数据库 '{self.database}' 中")
            sys.exit(1)
        
        # 获取表信息
        row_count = self.get_table_row_count(table_name)
        columns = self.get_table_columns(table_name)
        column_comments = self.get_column_comments(table_name) if include_comments else {}
        
        if not columns:
            print(f"✗ 错误: 无法获取表 '{table_name}' 的列信息")
            sys.exit(1)
        
        print(f"✓ 表 '{table_name}' 信息:")
        print(f"  - 总行数: {row_count:,}")
        print(f"  - 列数: {len(columns)}")
        print(f"  - 列名: {', '.join(columns)}")
        
        # 显示有注释的字段
        commented_columns = [col for col in columns if column_comments.get(col)]
        if commented_columns:
            print(f"  - 有注释的字段数: {len(commented_columns)}/{len(columns)}")
        
        # 确定实际导出的行数
        export_count = limit if limit is not None and limit < row_count else row_count
        if limit is not None:
            print(f"  - 将导出: {export_count:,} 行")
        
        # 构建查询SQL
        if limit is not None:
            query = f"SELECT * FROM `{table_name}` LIMIT {limit}"
        else:
            query = f"SELECT * FROM `{table_name}`"
        
        # 打开CSV文件并写入数据
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.writer(csvfile)
                
                # 写入表头
                writer.writerow(columns)
                
                # 写入字段注释行（如果存在注释且启用）
                if include_comments and any(column_comments.get(col) for col in columns):
                    comment_row = [column_comments.get(col, "") for col in columns]
                    writer.writerow(comment_row)
                
                # 执行查询并写入数据
                with self.connection.cursor() as cursor:
                    cursor.execute(query)
                    
                    exported = 0
                    while True:
                        rows = cursor.fetchmany(batch_size)
                        if not rows:
                            break
                        
                        for row in rows:
                            # 按照列顺序提取值
                            values = [self.convert_value(row.get(col, "")) for col in columns]
                            writer.writerow(values)
                            exported += 1
                            
                            # 显示进度
                            if exported % 1000 == 0:
                                print(f"  已导出: {exported:,} / {export_count:,} 行", end='\r')
                    
                    print(f"  已导出: {exported:,} / {export_count:,} 行")
            
            print(f"✓ 成功导出到文件: {output_path.absolute()}")
            print(f"  - 文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
            
        except Exception as e:
            print(f"✗ 导出过程中出错: {e}")
            if output_path.exists():
                output_path.unlink()  # 删除不完整的文件
            sys.exit(1)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="将SQL数据库中的指定表导出为CSV文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "table_name",
        help="要导出的表名"
    )
    
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="输出CSV文件路径（默认: <table_name>.csv）"
    )
    
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="数据库主机地址（默认: 127.0.0.1）"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=9030,
        help="数据库端口（默认: 9030）"
    )
    
    parser.add_argument(
        "--user", "-u",
        default="root",
        help="数据库用户名（默认: root）"
    )
    
    parser.add_argument(
        "--password", "-P",
        default="",
        help="数据库密码（默认: 空）"
    )
    
    parser.add_argument(
        "--database", "-d",
        default="database_main",
        help="数据库名称（默认: database_main）"
    )
    
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="限制导出的行数（用于测试，默认: 导出全部）"
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=10000,
        help="批量查询的大小（默认: 10000）"
    )
    
    parser.add_argument(
        "--no-comments",
        action="store_true",
        help="不导出字段注释"
    )
    
    args = parser.parse_args()
    
    # 确定输出文件路径
    if args.output:
        output_file = args.output
    else:
        output_file = f"{args.table_name}.csv"
    
    # 创建导出器并执行导出
    exporter = TableExporter(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        database=args.database
    )
    
    try:
        exporter.connect()
        exporter.export_table(
            table_name=args.table_name,
            output_file=output_file,
            limit=args.limit,
            batch_size=args.batch_size,
            include_comments=not args.no_comments
        )
    finally:
        exporter.close()


if __name__ == "__main__":
    main()

