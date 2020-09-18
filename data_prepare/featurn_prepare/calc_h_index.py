import pandas as pd
from tqdm import tqdm

from ..util import db_utils as db


def get_h_index(papers: pd.DataFrame) -> int:
    paper_count = papers.count()['citedby']
    papers = papers.sort_values(by='citedby')

    for i in range(paper_count):
        if papers['citedby'].iloc[i] >= (paper_count - i):
            return paper_count - i
    return 0


def get_i10_index(papers: pd.DataFrame) -> int:
    papers = papers.loc[papers['citedby'] >= 10]
    return papers.count()['citedby']


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


def read_author_data_frame(fellow: dict, cut=True):

    fellow_id = fellow['id']
    select_year = fellow['select_year']
    if select_year <= 0:
        select_year = 2020

    papers = db.list_author_papers(fellow_id, properties=['id', 'citedby', 'year'])

    # filter by paper year
    year_filter = filter_paper_by_year_gen(select_year, cut)
    papers = filter(year_filter, papers)

    # fill paper citeby
    papers = map(fill_paper_cite, papers)
    papers = list(papers)

    if len(papers) == 0:
        return None, None, None

    # find first pub year
    years = list(map(lambda p: p['year'], papers))
    first_pub_year = min(years)
    last__pub_year = max(years)

    return pd.DataFrame(data=papers), first_pub_year, last__pub_year


def main():
    for fellow in tqdm(db.list_fellow_with_dynamic_cols_is_null(dynamic_cols=['h_index', 'i10_index'])):
        papers, first_pub_year, last__pub_year = read_author_data_frame(fellow, cut=False)
        if papers is None:
            continue

        last_h_index = 0
        last_i10_index = 0
        for current_year in range(first_pub_year, now + 1):

            if current_year > last__pub_year:
                h_index = last_h_index
                i10_index = last_i10_index
            else:
                h_index = get_h_index(papers.loc[papers['year'] <= current_year])
                i10_index = get_i10_index(papers.loc[papers['year'] <= current_year])
            print("fellow-{} current_year={} h-index={} i10-index={}".format(
                fellow['id'], current_year, h_index, i10_index))

            db.update_table_by_condition(
                condition_dict={'id': fellow['id'], 'current_year': current_year},
                key_value_dict={'h_index': h_index, 'i10_index': i10_index},
                table=dynamic_fellow_table
            )
            last_h_index = h_index
            last_i10_index = i10_index


if __name__ == '__main__':
    now = 2020
    fellow_table = 'full_fellow_feature'
    dynamic_fellow_table = 'dynamic_full_fellow_feature_extend'
    paper_table = 'p_paper'
    db.open_connection()
    main()
    db.close_connection()
