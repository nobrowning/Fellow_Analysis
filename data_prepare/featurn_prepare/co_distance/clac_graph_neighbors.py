from .. import db_utils as db
from scipy.spatial.distance import cosine
from typing import Dict, Tuple, List, Set
import os
import re
from tqdm import tqdm
import networkx as nx
from collections import Counter


# 获得一个节点的邻居节点
def get_node_neighbors(graph, node_id: int) -> Tuple[Set[int], Counter]:
    neighbor_infos = dict(graph[node_id])
    neighbor_ids = neighbor_infos.keys()
    neighbor_weight = Counter()
    for neighbor_id, edge_property in neighbor_infos.items():
        neighbor_weight[neighbor_id] = edge_property['weight']

    return set(neighbor_ids), neighbor_weight


# 从邻接节点集中统计fellow个数(count)和合作次数(weight)
def get_fellow_count_and_weight(neighbor_ids: Set[int], neighbor_weight_dict: Counter, current_fellow_ids: Set[int]):
    neighbor_fellow_ids = current_fellow_ids & neighbor_ids
    fellow_count = len(neighbor_fellow_ids)
    fellow_weight = sum([neighbor_weight_dict[neighbor_id] for neighbor_id in list(neighbor_fellow_ids)])
    return fellow_count, fellow_weight


def main():
    for current_year in tqdm(range(first_select_year, 2020 + 1)):
        current_fellows = db.list_fellow_by_select_year(properties=['id'], select_year=current_year,
                                                        fellow_type=fellow_type)
        current_fellow_ids = set([int(f['id']) for f in current_fellows])

        edge_list_file = os.path.join(edge_list_file_dir, fellow_type, 'graph-{}.edgelist'.format(current_year))
        graph = nx.read_edgelist(path=edge_list_file, nodetype=int, data=(('weight', int),), create_using=nx.DiGraph())
        graph = graph.to_undirected()
        candidate_ids = list(graph.nodes())

        for candidate_id in candidate_ids:
            all_neighbor_ids = set()
            all_neighbor_weights = Counter()
            next_level_neighbor_ids = {candidate_id}
            result: Dict[str, int] = dict()
            for current_search_level in range(search_level):
                new_next_level_neighbor_ids = set()
                # calculate each neighbor's neighbor
                for neighbor_id in list(next_level_neighbor_ids):
                    one_node_neighbor_ids, one_node_neighbor_weights = get_node_neighbors(graph, neighbor_id)

                    # except current candidate
                    one_node_neighbor_ids = one_node_neighbor_ids - {candidate_id}

                    # append
                    new_next_level_neighbor_ids = new_next_level_neighbor_ids | one_node_neighbor_ids
                    all_neighbor_ids = all_neighbor_ids | one_node_neighbor_ids
                    all_neighbor_weights += one_node_neighbor_weights
                next_level_neighbor_ids = new_next_level_neighbor_ids

                # calculate on current level
                fellow_count, fellow_weight = get_fellow_count_and_weight(all_neighbor_ids, all_neighbor_weights,
                                                                          current_fellow_ids)
                result['{}_hop_fellow_count'.format(current_search_level + 1)] = fellow_count
                result['{}_hop_fellow_weight'.format(current_search_level + 1)] = fellow_weight

            db.update_table_by_condition(condition_dict={'id': candidate_id, 'current_year': current_year},
                                         key_value_dict=result,
                                         table='dynamic_full_fellow_feature_extend')


if __name__ == '__main__':
    search_level = 3
    first_select_year = 0
    distance_col_name = ''
    edge_list_file_dir = os.path.join('..', '..', '..', 'data', 'graph', 'edgelist')
    for fellow_type in ['acm', 'ieee']:
        if fellow_type == 'acm':
            first_select_year = 1994
            distance_col_name = 'co_distance_acm'
        elif fellow_type == 'ieee':
            first_select_year = 1953
            distance_col_name = 'co_distance_ieee'
        pattern = re.compile(r'graph-(\d+)\.edgelist')
        db.open_connection()
        main()
        db.close_connection()
