from langchain.tools import tool, ToolRuntime
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import json

db_user_name = "root"
db_host = "127.0.0.1"
db_pwd = ""
port = 9030
db_name = "database_main"

db_engine = create_engine(f"mysql+pymysql://{db_user_name}:{db_pwd}@{db_host}:{port}/{db_name}")

with open(f'./T3/data/mschema_{db_name}.json', 'r', encoding='utf-8') as f:
    mschema_json = json.load(f)

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
                return json.dumps({"status": "ok", "rows": rows}, ensure_ascii=False)

            connection.commit()
            return json.dumps(
                {"status": "ok", "rows": [], "rowcount": result.rowcount},
                ensure_ascii=False,
            )
    except SQLAlchemyError as exc:
        return json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False)
    


#=== Prompt 使用的工具 ===#

def get_mschema(table_name: str) -> str:
    """Retrieve the M-Schema for a specific table."""
    o = mschema_json.get("tables", {}).get(table_name, {})
    n = {"table_name": table_name}
    n.update(o)
    return n