from tools.tools4sql import get_mschema
import json
from toon import encode
from transformers import AutoTokenizer

with open("T3/data/final_dataset.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)

with open("T3/data/common_knowledge.md", "r", encoding="utf-8") as f:
    common_knowledge = f.read()

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True)

for t in json_data:
    
    prompt = f"""你是一名StarRocks mysql 4.0.0专家，现在需要阅读并理解下面的[数据库schema]描述，以及可能用到的[参考信息]，并运用StarRocks mysql 4.0.0知识生成sql语句回答[用户问题]。
[用户问题]: {t["question"]}  
[数据库schema]: {encode([get_mschema(i) for i in t["table_list"]])}
[参考信息]: {t["knowledge"]}
[通用知识]: {common_knowledge}
[涉及的表名列表]: {t["table_list"]}  
"""

    print(f"{prompt}")
    print(f"Token count: {len(tokenizer.encode(prompt))}")

    with open(f"T3/script/prompt/input/V1/{t['sql_id']}.txt", "w", encoding="utf-8") as f:
        f.write(prompt)