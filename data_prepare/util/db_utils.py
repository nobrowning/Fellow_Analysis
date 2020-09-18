import datetime
from typing import List, Tuple, Optional, Dict
import pymysql
from pymysql import Connection
from pymysql.cursors import SSDictCursor
import numpy


host = '127.0.0.1'
user_name = 'root'
user_password = 'root'
db_name = 'fellow_20_mag'

connection: Connection
cursor: SSDictCursor


def open_connection() -> pymysql.Connection:
    global connection
    global cursor
    connection = Connection(host, user_name, user_password, db_name, charset='utf8',
                            cursorclass=SSDictCursor)
    cursor = connection.cursor()


def close_connection():
    global connection
    global cursor
    if connection is not None:
        cursor.close()
        connection.close()

        connection = None
        cursor = None


def list_fellow(properties: list, table='full_fellow_feature') -> List:

    if len(properties) == 0:
        properties.append("*")
    properties_str = ','.join(properties)
    sql = "SELECT " + properties_str + " FROM " + table + \
          " WHERE (gsLink IS NOT NULL AND gsLink <> 'null') or ms_id IS NOT NULL"
    fellow_list = []
    cursor.execute(sql)
    fellow_list = cursor.fetchall()
    return fellow_list


def list_fellow_dynamic_null(properties: list, table='full_fellow_feature') -> List:
    global cursor
    if len(properties) == 0:
        properties.append("*")
    properties_str = ','.join(properties)
    sql = "SELECT " + properties_str + " FROM " + table + \
          " WHERE ms_id IS NOT NULL " \
          "AND id NOT IN (SELECT distinct(id) FROM dynamic_full_fellow_feature_extend) " \
          "AND finished=0"
    fellow_list = []

    cursor.execute(sql)
    fellow_list = cursor.fetchall()

    return fellow_list


def list_fellow_by_select_year(properties: list, select_year: int, fellow_type='ieee') -> List[Dict[str, object]]:
    if len(properties) == 0:
        properties.append("*")
    properties_str = ','.join(properties)
    sql = "SELECT " + properties_str + " FROM full_fellow_feature " \
          "WHERE type1=%s AND type2='fellow' AND select_year<=%s"
    fellow_list = []

    global cursor
    cursor.execute(sql, (fellow_type, select_year))
    fellow_list = cursor.fetchall()
    return fellow_list


def list_fellow_with_dynamic_cols_is_null(dynamic_cols: list,
                                          fellow_types=['ieee', 'acm', 'aminer'],
                                          properties=['*'],
                                          dynamic_table='dynamic_full_fellow_feature_extend',
                                          static_feature_table='full_fellow_feature',) -> List:

    properties_str = ','.join(properties)
    conditions_str = ' OR '.join(['{} IS NULL'.format(col) for col in dynamic_cols])

    fellow_type_str = 'type1 IN ({})'.format(', '.join(['%s'] * len(fellow_types)))

    sql = "SELECT " + properties_str + " FROM " + static_feature_table + " WHERE id IN ( SELECT DISTINCT(id) FROM " \
          + dynamic_table + " WHERE " + conditions_str + " AND " + fellow_type_str + " )"

    global cursor
    cursor.execute(sql, tuple(fellow_types))
    fellows = cursor.fetchall()

    return fellows


def list_author_papers(author_id: int, properties: list, ap_table='p_paper_fellow_relation', p_table='p_paper'):

    if len(properties) == 0:
        properties.append("*")
    properties_str = ','.join(properties)
    sql = "SELECT " + properties_str + " FROM " + p_table + \
          " WHERE id IN ( " \
          "SELECT paper_id FROM " + ap_table + " WHERE fellow_id = %s)"

    papers = []
    global cursor
    cursor.execute(sql, author_id)
    papers = cursor.fetchall()

    print('# papers: {}'.format(len(papers)))
    return papers


def update_table_by_id(_id: str, key_value_dict: dict, table='full_fellow_feature'):
    sql_set_part = ""
    values = []
    for key, value in key_value_dict.items():
        sql_set_part += key + "=%s,"
        values.append(value)

    sql = "UPDATE " + table + " SET " + sql_set_part[:-1] + " WHERE id=%s"
    values.append(_id)

    global cursor
    cursor.execute(sql, tuple(values))
    connection.commit()


def update_table_by_condition(condition_dict: dict, key_value_dict: dict, table='dynamic_full_fellow_feature'):
    sql_set_part = ""
    values = []
    for key, value in key_value_dict.items():
        if type(value) is numpy.float64:
            value = float("{0:.4f}".format(value))
        if type(value) is numpy.int64:
            value = int(value)
        sql_set_part += key + "=%s,"
        values.append(value)

    sql_condition_part = ' AND '.join(['{}=%s'.format(key) for key in condition_dict.keys()])
    values += condition_dict.values()

    sql = "UPDATE " + table + " SET " + sql_set_part[:-1] + " WHERE " + sql_condition_part

    global cursor
    cursor.execute(sql, tuple(values))
    connection.commit()


def insert_fellow_features(params: dict, table):
    key_list = []
    value_list = []
    for key, value in params.items():
        if type(value) is numpy.float64:
            value = float("{0:.3f}".format(value))
        if type(value) is numpy.int64:
            value = int(value)
        key_list.append(key)
        value_list.append(value)
    dt = datetime.datetime.now()

    global cursor
    sql = "INSERT INTO " + table + " (" + ", ".join(key_list) + ") VALUE (" + ','.join(['%s']*len(key_list)) + ")"
    cursor.execute(sql, tuple(value_list))
    connection.commit()

    return None


def list_paper_relation_with_filter(
        filter_year=1912,
        properties=['*'],
        relation_table='p_paper_fellow_relation',
        dynamic_table='dynamic_full_fellow_feature'):
    properties_str = ','.join(properties)
    sql = "SELECT " + properties_str + " FROM " + relation_table + \
          " WHERE fellow_id IN ( " \
          "SELECT DISTINCT(id) FROM " + dynamic_table + " ) AND year >=" + str(filter_year)

    global cursor
    cursor.execute(sql)
    relations = cursor.fetchall()
    return relations


def list_double_fellow(properties=['*']):
    properties_str = ','.join(properties)
    sql = "SELECT " + properties_str + " FROM full_fellow_feature" \
          " WHERE ms_id IN ( SELECT ms_id FROM full_fellow_feature" \
          " WHERE ms_id IS NOT NULL" \
          " GROUP BY ms_id" \
          " HAVING COUNT(*) > 1)"

    global cursor
    cursor.execute(sql)
    relations = cursor.fetchall()
    return relations
