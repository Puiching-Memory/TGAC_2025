import json
from transformers import AutoTokenizer
from toon_format import encode

def get_mschema(table_name: str) -> str:
    """Retrieve the M-Schema for a specific table."""
    with open(f'./T3/data/mschema_database_main.json', 'r', encoding='utf-8') as f:
        mschema_json = json.load(f)

    o = mschema_json.get("tables", {}).get(table_name, {})
    n = {"table_name": table_name}
    n.update(o)
    return n

with open("T3/data/final_dataset.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True)
# [schema信息]: {encode([get_mschema(i) for i in t["table_list"]])}

for t in json_data:
    if t.get("golden_sql"):continue

    prompt = f"""你是一名StarRocks mysql 4.0.0专家, 请你生成sql语句回答[用户问题]。
[用户问题]: {t["question"]}  
[涉及表名]: {t["table_list"]} 
[提示]: {t["knowledge"] or "无"}
"""

    print(f"{prompt}")
    print(f"Token count: {len(tokenizer.encode(prompt))}")

    with open(f"T3/script/prompt/input/V1/{t['sql_id']}.txt", "w", encoding="utf-8") as f:
        f.write(prompt)