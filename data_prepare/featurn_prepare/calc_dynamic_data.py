# 与原版的区别是，所有学者的数据都计算到2020年。
# 原版只计算到到学者的最后出版年
import math
from typing import Optional
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
from tqdm import tqdm
import ast

from ..util import db_utils as db


def append_cum_column(df: pd.DataFrame, src_field: str, tar_field: str, cnt_field: str) -> pd.DataFrame:
    pub_list = df[src_field]
    cum = 0
    cum_list = []
    t = 0
    t_list = []
    count_list = []
    # begin = False
    for v in pub_list:
        t += 1
        t_list.append(t)
        cum += v
        cum_list.append(cum)
        count_list.append(v)
    # print(cum_list)
    df = pd.DataFrame({'t': pd.Series(t_list), tar_field: pd.Series(cum_list), cnt_field: pd.Series(count_list)})
    # print(df)
    return df


def calc_alpha(df: pd.DataFrame, col: str):
    dt_1 = pd.DataFrame({'x': np.log(df['t']), 'y': np.log(df[col])})
    if dt_1.size == 0:
        return 1, 1, 1
    sm_result = sm.ols(formula='y ~ x', data=dt_1).fit()
    # print(sm_result.params)
    alpha = round(sm_result.params['x'], 2)
    std = round(sm_result.bse['x'], 2)
    A = round(np.exp(sm_result.params['Intercept']), 5)
    if math.isnan(alpha) or math.isinf(alpha) or alpha < 1:
        alpha = 1
    if math.isnan(std) or math.isinf(std):
        std = 1
    if math.isnan(A) or math.isinf(A):
        A = 1
    return alpha, A, std


def filter_paper_by_year_gen(select_year: int, cut: bool):
    def filter_paper_by_year(paper: dict) -> bool:
        year = paper['year']
        if year is None or select_year - year >= 60:
            return False
        if cut and year > select_year:
            return False
        return True
    return filter_paper_by_year


def fill_paper_cite(paper: dict) -> dict:
    cited_by = paper['citedby']
    if cited_by is None or cited_by < 0:
        paper['citedby'] = 0
    return paper


def read_author_data_frame(fellow: dict, cut=True) -> Optional[pd.DataFrame]:

    fellow_id = fellow['id']
    select_year = fellow['select_year']
    if select_year <= 0:
        select_year = 2020

    papers = db.list_author_papers(fellow_id, properties=[], ap_table="p_paper_fellow_relation", p_table="p_paper")

    # filter by paper year
    year_filter = filter_paper_by_year_gen(select_year, False)
    papers = filter(year_filter, papers)

    # fill paper citeby
    papers = map(fill_paper_cite, papers)
    papers = list(papers)

    papers = list(papers)
    if len(papers) == 0:
        return None

    # find first pub year
    years = list(map(lambda p: p['year'], papers))
    first_pub_year = min(years)
    last__pub_year = max(years)

    year_num = select_year - first_pub_year + 1
    if year_num < 8:
        return None

    if fellow['cite_by_year'] is None:
        return None
    cite_dict = ast.literal_eval(fellow['cite_by_year'])
    if len(cite_dict.items()) == 0:
        return None

    # 统计截止年份
    end_year = (select_year if cut else now) + 1

    print("{}-{}={}.".format(fellow['id'], fellow['name'], first_pub_year))
    fellow['first_pub_year'] = first_pub_year

    db.update_table_by_id(fellow_id, {'first_pub_year': first_pub_year}, table=fellow_table)

    df = pd.DataFrame(data=papers, columns=['id', 'year', 'citedby'])
    # print(df.head())

    pub_year_count = df.groupby(['year'])['id'].count()
    pub_year_count = pub_year_count.to_frame()
    pub_year_count = pub_year_count.reindex(pd.Index(np.arange(first_pub_year, end_year, 1), name='year'), fill_value=0)

    cite_year_count = pd.DataFrame.from_dict(data=cite_dict, orient='index')
    try:
        cite_year_count.columns = ['citedby']
    except ValueError as e:
        pass
    cite_year_count = cite_year_count.reindex(pd.Index(np.arange(first_pub_year, end_year, 1), name='year'),
                                              fill_value=0)
    pub_cum_df = append_cum_column(pub_year_count, 'id', 'pub_cum', 'pub_num')
    cite_cum_df = append_cum_column(cite_year_count, 'citedby', 'cite_cum', 'cite_num')
    result_df = pd.concat([pub_cum_df, cite_cum_df], axis=1)
    result_df = pd.concat([result_df, pd.Series([x for x in range(first_pub_year, end_year)], name='year')], axis=1)
    result_df.columns = ['t', 'pub_cum', 'pub_num', 't2', 'cite_cum', 'cite_num', 'year']
    # print(result_df)
    return result_df


def main():
    for fellow in tqdm(db.list_fellow_dynamic_null(properties=[])):
        df = read_author_data_frame(fellow, cut=False)
        if df is None:
            db.update_table_by_id(fellow['id'], {'finished': True})
            continue
        
        for i in range(df['t'].iloc[-1]):
            pub_num = df['pub_num'].iloc[i]
            pub_cum = df['pub_cum'].iloc[i]
            print('pub_num={}'.format(pub_num))

            last_year = df['year'].iloc[-1]
            cur_year = df['year'].iloc[i]
            print("# papers[<={}] = {}".format(last_year, pub_cum))
            year_num = len(df)
            pub_year_avg = pub_cum / (i+1)
            pub_year_avg = round(pub_year_avg, 3)
            print('# paper_per_year = {}'.format(pub_year_avg))

            cite_num = df['cite_num'].iloc[i]
            cite_cum = df['cite_cum'].iloc[i]
            cite_per_year = cite_cum / (i+1)
            cite_per_year = round(cite_per_year, 3)
            print("# citation[<={}] = {}".format(last_year, cite_cum))
            print('# cite_per_year = {}'.format(cite_per_year))

            pub_alpha, _, _ = calc_alpha(df.loc[df['year'] <= cur_year], 'pub_cum')
            cite_alpha, _, _ = calc_alpha(df.loc[(df['year'] <= cur_year) & (df['cite_cum'] > 0)], 'cite_cum')

            db.insert_fellow_features(
                {'id': fellow['id'], 'name': fellow['name'], 'type1': fellow['type1'], 'type2': fellow['type2'],
                 'gender': fellow['gender'], 'current_year': cur_year, 'select_year': fellow['select_year'],
                 'pub_num': pub_num, 'pub_cum': pub_cum, 'pub_per_year': pub_year_avg, 'cite_num': cite_num,
                 'cite_cum': cite_cum, 'cite_per_year': cite_per_year, 'pub_alpha': pub_alpha, 'cite_alpha': cite_alpha,
                 'first_pub_year': fellow['first_pub_year']},
                table=dynamic_fellow_table)

        db.update_table_by_id(fellow['id'], {'finished': True})
            # break


if __name__ == '__main__':
    now = 2020
    fellow_table = 'full_fellow_feature'
    dynamic_fellow_table = 'dynamic_full_fellow_feature_extend'
    paper_table = 'p_paper'
    db.open_connection()
    main()
    db.close_connection()
