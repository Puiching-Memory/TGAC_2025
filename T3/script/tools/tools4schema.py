import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../lib/M-Schema'))

from sqlalchemy import create_engine
from schema_engine import SchemaEngine

db_user_name = "root"
db_host = "127.0.0.1"
db_pwd = ""
port = 9030
db_name = "database_main"

db_engine = create_engine(f"mysql+pymysql://{db_user_name}:{db_pwd}@{db_host}:{port}/{db_name}")

schema_engine = SchemaEngine(engine=db_engine, db_name=db_name)
mschema = schema_engine.mschema
mschema_str = mschema.to_mschema()
print(mschema_str)
mschema.save(f'./T3/data/mschema_{db_name}.json')
