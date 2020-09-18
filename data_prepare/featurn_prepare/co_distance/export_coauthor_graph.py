from .. import db_utils as db
from typing import Tuple, List, Dict, Set
import itertools
from tqdm import tqdm
import pandas as pd
import os


def get_double_fellow_id_pair() -> Set[Tuple[int]]:
    fellow_groups: Dict[int, List[int]] = dict()
    for fellow in db.list_double_fellow(properties=['id', 'ms_id']):
        fellow_id = fellow['id']
        ms_id = fellow['ms_id']

        if ms_id in fellow_groups.keys():
            fellow_groups[ms_id].append(fellow_id)
        else:
            fellow_groups[ms_id] = [fellow_id, ]

    fellow_id_pairs = set()
    for fellow_id_list in fellow_groups.values():
        fellow_id_list.sort()
        fellow_id_pairs.add(tuple(fellow_id_list))

    return fellow_id_pairs


def update_paper_groups(current_relations: pd.DataFrame, paper_groups: Dict[str, List[int]]):
    for index, relation in current_relations.iterrows():

        paper_id = relation['paper_id']
        fellow_id = relation['fellow_id']

        if fellow_id not in fellow_ids:
            continue

        if paper_id in paper_groups.keys():
            paper_groups[paper_id].append(fellow_id)
        else:
            paper_groups[paper_id] = [fellow_id, ]


def get_edges(paper_groups: Dict[int, List[int]]) -> Dict[Tuple[int, int], int]:
    edges: Dict[Tuple[int, int], int] = dict()

    for co_authors in paper_groups.values():

        if len(co_authors) < 2:
            continue

        for co_author_1, co_author_2 in itertools.combinations(co_authors, r=2):

            # sort
            if co_author_1 > co_author_2:
                t = co_author_2
                co_author_2 = co_author_1
                co_author_1 = t

            # weight count
            edge_key = (co_author_1, co_author_2)

            # skip double fellow
            if edge_key in double_fellow_id_pair:
                continue

            if edge_key in edges.keys():
                edges[edge_key] += 1
            else:
                edges[edge_key] = 1
    return edges


def main():
    pub_start_year = 1912
    this_year = 2020
    relations = db.list_paper_relation_with_filter(filter_year=pub_start_year,
                                                   properties=['paper_id', 'fellow_id', 'year'])
    relations_df = pd.DataFrame(relations)

    paper_groups: Dict[int, List[int]] = dict()
    years_range_process = tqdm(range(pub_start_year, this_year + 1))
    for current_year in years_range_process:
        years_range_process.set_description("current year: {}".format(current_year))
        current_relations = relations_df.loc[relations_df['year'] == current_year]
        update_paper_groups(current_relations, paper_groups)
        edges = get_edges(paper_groups)
        with open(os.path.join(edge_list_file_dir, fellow_type, "graph-{}.edgelist".format(current_year)), 'w') as f:
            for (node1, node2), weight in edges.items():
                print(node1, node2, weight, sep='\t', end='\n', file=f)


if __name__ == '__main__':
    db.open_connection()
    edge_list_file_dir = os.path.join('..', '..', '..', 'data', 'graph', 'edgelist')
    for fellow_type in ['acm', 'ieee']:
        fellow_ids = db.list_fellow_with_dynamic_cols_is_null(
            dynamic_cols=['co_dictance'],
            fellow_types=[fellow_type, 'aminer'],
            properties=['id'])
        fellow_ids = set([f['id'] for f in fellow_ids])
        double_fellow_id_pair = get_double_fellow_id_pair()
        main()
    db.close_connection()
