import numpy as np
import pymysql
import random
from typing import List, Tuple
from tqdm import tqdm
import pandas as pd
import pickle
from pathlib import Path
import os

# prepare classification input data for reg-Fellow and other classification baseline models


def find_field_vector(fellow_id, year, first_pub_year):
    if (fellow_id, year) in title_embedding_dict.keys():
        return title_embedding_dict[(fellow_id, year)]
    else:
        for y in range(year - 1, first_pub_year - 1, -1):
            if (fellow_id, y) in title_embedding_dict.keys():
                return title_embedding_dict[(fellow_id, y)]
    return np.zeros([8, ])


def fellow_data_2_np_array(fellow_data: pd.DataFrame, first_pub_year: int) -> np.ndarray:
    fellow_data = fellow_data.sort_values(by=['current_year'])
    fellow_id = fellow_data.iloc[0]['id']

    multi_year_features = []
    title_embs = []
    for index, one_year_fellow_data in fellow_data.iterrows():
        current_year = one_year_fellow_data['current_year']
        if current_year > cut_year:
            continue

        one_year_feature = []
        if 'male' == one_year_fellow_data['gender']:
            one_year_feature.append(1)
        else:
            one_year_feature.append(0)

        for feature_name in feature_cols:
            feature_value = one_year_fellow_data[feature_name]
            one_year_feature.append(feature_value)
        multi_year_features.append(one_year_feature)
        title_embs.append(find_field_vector(fellow_id, current_year, first_pub_year))

    multi_year_features = np.array(multi_year_features, dtype=np.float32)
    title_embs = np.array(title_embs, dtype=np.float32)

    multi_year_features = np.concatenate((multi_year_features, title_embs), axis=1)

    # fill zeros
    fill_year_count = first_pub_year - start_year
    filled_year_features = np.zeros((fill_year_count, multi_year_features.shape[1]), dtype=np.float32)

    return np.concatenate((filled_year_features, multi_year_features))


def main():
    fellow_scholar_ids = \
        all_fellow_data.loc[(all_fellow_data['type1'] == fellow_type)
                            & (all_fellow_data['type2'] == 'fellow')
                            & (all_fellow_data['select_year'] <= cut_year)
                            & (all_fellow_data['current_year'] > cut_year - 3)
                            & (all_fellow_data['current_year'] <= cut_year)
                            & (all_fellow_data['pub_num'] > 0)]
    fellow_scholar_ids = fellow_scholar_ids.sort_values(by='select_year', ascending=False)['id'].unique()
    fellow_scholar_ids = list(fellow_scholar_ids)

    if 'acm' == fellow_type:
        acm_non_fellow_scholar_ids = \
            standard_single_type_fellow_data.loc[(standard_single_type_fellow_data['type1'] == fellow_type)
                                                 & (standard_single_type_fellow_data['type2'] == 'fellow')
                                                 & (standard_single_type_fellow_data['select_year'] > cut_year)
                                                 | ((standard_single_type_fellow_data['type2'] != 'fellow')
                                                    & (standard_single_type_fellow_data['type1'] == fellow_type)), 'id'].unique()
        acm_non_fellow_scholar_ids = list(acm_non_fellow_scholar_ids)

        other_non_fellow_num = len(fellow_scholar_ids) - len(acm_non_fellow_scholar_ids)
        non_fellow_ids = acm_non_fellow_scholar_ids + other_non_fellow_scholar_ids[:other_non_fellow_num]
    else:
        non_fellow_ids = \
            standard_single_type_fellow_data.loc[(standard_single_type_fellow_data['type1'] == fellow_type)
                                                 & (standard_single_type_fellow_data['type2'] == 'fellow')
                                                 & (standard_single_type_fellow_data['select_year'] > cut_year)
                                                 | (standard_single_type_fellow_data['type2'] != 'fellow'), 'id'].unique()
        non_fellow_ids = list(non_fellow_ids)
        fellow_scholar_ids = fellow_scholar_ids[:len(non_fellow_ids)]

    features_and_targets: List[Tuple[np.ndarray, int]] = []
    for scholar_id in tqdm(fellow_scholar_ids):
        if scholar_id not in saved_fellow_ids:
            continue

        fellow_data = standard_single_type_fellow_data.loc[standard_single_type_fellow_data['id'] == scholar_id]

        first_pub_year = int(fellow_data.iloc[0]['first_pub_year'])
        one_fellow_features = fellow_data_2_np_array(fellow_data, first_pub_year)
        one_fellow_target = 1
        features_and_targets.append((one_fellow_features, one_fellow_target))

    for scholar_id in tqdm(non_fellow_ids):
        if scholar_id not in saved_fellow_ids:
            continue

        fellow_data = standard_single_type_fellow_data.loc[standard_single_type_fellow_data['id'] == scholar_id]

        first_pub_year = int(fellow_data.iloc[0]['first_pub_year'])
        one_fellow_features = fellow_data_2_np_array(fellow_data, first_pub_year)
        one_fellow_target = 0
        features_and_targets.append((one_fellow_features, one_fellow_target))

    random.shuffle(features_and_targets)

    train_features = []
    train_targets = []
    for feature, label in features_and_targets:
        train_features.append(feature)
        train_targets.append(label)

    train_features = np.array(train_features, dtype=np.float32)
    train_targets = np.array(train_targets, dtype=np.int32)

    np.save(os.path.join(data_dir_path, 'all_features.npy'), train_features)
    np.save(os.path.join(data_dir_path, 'all_targets.npy'), train_targets)


if __name__ == '__main__':
    fellow_types = ['acm', 'ieee']

    for fellow_type in fellow_types:

        start_year = 1936
        if 'acm' == fellow_type:
            start_year = 1936
            cut_year_range = range(2015, 2019 + 1)
        elif 'ieee' == fellow_type:
            start_year = 1919
            cut_year_range = range(2016, 2020 + 1)
        feature_cols = ['org_id', 'pub_num', 'pub_alpha', 'cite_num', 'cite_cum', 'cite_alpha', 'h_index',
                        'i10_index', 'co_distance_{}'.format(fellow_type), 'pub_per_year', 'cite_per_year',
                        'continue_year']

        title_embedding_dict = pickle.load(open(os.path.join('..', 'data', 'tech_area_{}.pkl'.format(fellow_type)), 'rb'))
        saved_fellow_ids = set(map(lambda k: k[0], title_embedding_dict.keys()))

        all_fellow_data = pd.read_csv(os.path.join('..', 'data', 'dynamic_fellow_data.csv'.format(fellow_type)))
        standard_single_type_fellow_data = \
            pd.read_csv(os.path.join('..', 'data', 'dynamic_fellow_data_{}.csv'.format(fellow_type)))

        other_non_fellow_scholar_ids: list
        if fellow_type == 'acm':
            other_non_fellow_scholar_ids = \
                standard_single_type_fellow_data.loc[(standard_single_type_fellow_data['type2'] != 'fellow')
                                                     & (standard_single_type_fellow_data['type1'] != 'acm'), 'id'].unique()
            other_non_fellow_scholar_ids = list(other_non_fellow_scholar_ids)
            random.shuffle(other_non_fellow_scholar_ids)

        cut_year: int
        for year in cut_year_range:
            cut_year = year
            data_dir_path = Path(os.path.join('..', 'data', 'cls_data', fellow_type, str(cut_year)))
            data_dir_path.mkdir(parents=True, exist_ok=True)
            print('{}: cut year: {}'.format(fellow_type, cut_year))
            main()
