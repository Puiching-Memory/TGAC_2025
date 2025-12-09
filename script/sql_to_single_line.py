#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将多行 SQL 转换为单行 SQL，并转义特殊字符
"""


def escape_sql_to_single_line(sql_text):
    """
    将多行 SQL 转换为单行，并转义特殊字符
    
    Args:
        sql_text: 多行 SQL 文本
        
    Returns:
        转义后的单行 SQL 字符串
    """
    # 转义特殊字符
    # 先转义反斜杠，避免后续转义时出现问题
    escaped = sql_text.replace('\\', '\\\\')
    # 转义换行符
    escaped = escaped.replace('\n', '\\n')
    # 转义回车符
    escaped = escaped.replace('\r', '\\r')
    # 转义制表符
    escaped = escaped.replace('\t', '\\t')
    # 转义双引号
    escaped = escaped.replace('"', '\\"')
    # 转义单引号（可选，取决于使用场景）
    # escaped = escaped.replace("'", "\\'")
    
    return escaped


def main():
    """主函数"""
    # 在这里输入你的多行 SQL
    sql = """WITH `active_users` AS
  (SELECT DISTINCT `a`.`iuserid`
   FROM `database_main`.`dws_argothek_oss_useractivity_df` `a`
   WHERE `a`.`statis_date` BETWEEN 20250518 AND 20250617),
     `users_with_all_items` AS
  (SELECT `g`.`iuserid`
   FROM `database_main`.`dwd_argothek_gearrecord_df` `g`
   WHERE `g`.`igoodsid` IN ('71d260ca-4bad-81f2-0298-44860341a7a6',
                            '48601754-442d-98cb-2109-3fb2075500ec',
                            'a9126263-439c-7be6-d668-52a6c0c0a36f',
                            '34ecc788-4034-71d2-a0ac-c398b8ecb2ae',
                            '56f98991-43f8-dcc8-189e-08b7ae6c42ad',
                            '8453f8ef-4c0b-46e2-8768-3e9e45c67a2c',
                            '06d90274-435f-3e32-9d57-26bc228ae2e5',
                            '8196dea4-4229-c951-13a9-088fb13b3512',
                            '3abc330f-444a-6dfd-5863-81856ba62624',
                            '041a43a4-46fe-dab7-c974-36bf91eab33d',
                            '5b863cb9-4e41-ec89-69ee-49a223d05ff1')
     AND `g`.`iuserid` IN
       (SELECT `iuserid`
        FROM `active_users`)
   GROUP BY `g`.`iuserid`
   HAVING COUNT(DISTINCT `g`.`igoodsid`) = 11)
SELECT
  (SELECT COUNT(*)
   FROM `active_users`) AS `近30天活跃玩家数`,

  (SELECT COUNT(*)
   FROM `users_with_all_items`) AS `近30天活跃且同时拥有奖池所有道具玩家数`
LIMIT 1000
"""
    
    # 转换为单行并转义特殊字符
    result = escape_sql_to_single_line(sql)
    
    # 输出结果
    print(result)


if __name__ == "__main__":
    main()

