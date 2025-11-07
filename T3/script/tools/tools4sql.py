from langchain.tools import tool, ToolRuntime
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import json
from decimal import Decimal
from datetime import datetime, date

# 自定义 JSON 编码器，处理 Decimal、datetime 等特殊类型
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            # 如果是整数，返回 int；否则返回 float
            return int(obj) if obj % 1 == 0 else float(obj)
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)

db_user_name = "root"
db_host = "127.0.0.1"
db_pwd = ""
port = 9030
db_name = "database_main"

db_engine = create_engine(f"mysql+pymysql://{db_user_name}:{db_pwd}@{db_host}:{port}/{db_name}")

#=== Agent 使用的工具 ===#

@tool
def run_sql_query(query: str) -> str:
    """Run a SQL query and return the result or error message."""
    try:
        with db_engine.connect() as connection:
            result = connection.execute(text(query))
            if result.returns_rows:
                columns = result.keys()
                rows = [dict(zip(columns, row)) for row in result.fetchall()]
                return json.dumps({"status": "ok", "rows": rows}, ensure_ascii=False, cls=CustomJSONEncoder)

            connection.commit()
            return json.dumps(
                {"status": "ok", "rows": [], "rowcount": result.rowcount},
                ensure_ascii=False,
                cls=CustomJSONEncoder
            )
    except SQLAlchemyError as exc:
        return json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False)
