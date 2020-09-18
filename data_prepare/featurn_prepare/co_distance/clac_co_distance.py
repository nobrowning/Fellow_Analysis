from .. import db_utils as db
from scipy.spatial.distance import cosine
from typing import Dict, Tuple, List
import os
import re
from tqdm import tqdm


def cosine_sim(arr1, arr2):
    return 1.0 - cosine(arr1, arr2)


def main():
    emb_files_dir = os.path.join(emb_root_dir, fellow_type)
    for emb_file_name in os.listdir(emb_files_dir):
        matcher = pattern.match(emb_file_name)
        current_year: int
        if matcher:
            current_year = int(matcher.group(1))
        else:
            continue

        query_select_year = current_year
        if current_year < first_select_year:
            query_select_year = first_select_year
        current_fellows = db.list_fellow_by_select_year(properties=['id'], select_year=query_select_year,
                                                        fellow_type=fellow_type)
        current_fellow_ids = set([f['id'] for f in current_fellows])
        current_fellow_len = len(current_fellow_ids)
        emb_file = open(os.path.join(emb_files_dir, emb_file_name), 'r')

        line1 = emb_file.readline().strip()
        tokens = line1.split(' ')
        node_cnt = int(tokens[0])
        vector_size = int(tokens[1])
        print('# current year:', current_year)
        print('# query select year:', query_select_year)
        print('# current fellows:', current_fellow_ids)
        print('# year:', current_year)
        print('# nodes:', node_cnt)
        print('# vector size:', vector_size)

        matrix = []
        candidate_id_2_index_map = {}
        candidate_index_2_id_map = {}

        cur_candidate_idx = 0
        for line in emb_file.readlines():
            tokens = line.strip().split(' ')
            candidate_id = int(tokens[0][1:])

            candidate_id_2_index_map[candidate_id] = cur_candidate_idx
            candidate_index_2_id_map[cur_candidate_idx] = candidate_id
            vec = [float(x) for x in tokens[1:]]
            matrix.append(vec)
            cur_candidate_idx += 1

        distance_cache: Dict[Tuple[int, int], float] = dict()
        process_bar = tqdm(candidate_id_2_index_map.items())
        for candidate_id, cur_candidate_idx in process_bar:
            c_score = 0.0
            for other_candidate_idx in range(len(matrix)):
                other_candidate_id = candidate_index_2_id_map[other_candidate_idx]
                if other_candidate_idx == cur_candidate_idx or other_candidate_id not in current_fellow_ids:
                    continue

                sim_score: float

                # index sort
                query_idx = cur_candidate_idx
                if other_candidate_idx > query_idx:
                    t = other_candidate_idx
                    other_candidate_idx = query_idx
                    query_idx = t

                if (other_candidate_idx, query_idx) not in distance_cache.keys():
                    sim_score = 1 - cosine_sim(matrix[query_idx], matrix[other_candidate_idx])
                    distance_cache[(other_candidate_idx, query_idx)] = sim_score
                else:
                    sim_score = distance_cache[(other_candidate_idx, query_idx)]
                c_score += sim_score
            c_score = round(c_score / (current_fellow_len - 1), 4)
            process_bar.set_description('{}:{}={}'.format(current_year, candidate_id, c_score))
            db.update_table_by_condition(condition_dict={'id': candidate_id, 'current_year': current_year},
                                         key_value_dict={distance_col_name: c_score},
                                         table='dynamic_full_fellow_feature_extend')


if __name__ == '__main__':
    first_select_year = 0
    distance_col_name = ''
    emb_root_dir = os.path.join('..', '..', '..', 'data', 'graph', 'emb')
    for fellow_type in ['acm', 'ieee']:
        if fellow_type == 'acm':
            first_select_year = 1994
            distance_col_name = 'co_distance_acm'
        elif fellow_type == 'ieee':
            first_select_year = 1952
            distance_col_name = 'co_distance_ieee'
        pattern = re.compile(r'coauthor-(\d+)\.emb')
        db.open_connection()
        main()
        db.close_connection()
