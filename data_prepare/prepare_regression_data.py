import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
from pathlib import Path
import os

# prepare regression input data for reg-Fellow and other regression baseline models


def find_field_vector(fellow_id, year, first_pub_year):
    if (fellow_id, year) in title_embedding_dict.keys():
        return title_embedding_dict[(fellow_id, year)]
    else:
        for y in range(year - 1, first_pub_year - 1, -1):
            if (fellow_id, y) in title_embedding_dict.keys():
                return title_embedding_dict[(fellow_id, y)]
    return np.zeros([8, ])


def fellow_data_2_np_array(fellow_data: pd.DataFrame, first_pub_year: int, extract_year: int)\
        -> np.ndarray:
    fellow_data = fellow_data.sort_values(by=['current_year'])
    fellow_id = fellow_data.iloc[0]['id']

    multi_year_features = []
    title_embeddings = []
    for index, one_year_fellow_data in fellow_data.iterrows():
        current_year = one_year_fellow_data['current_year']
        if current_year > extract_year:
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
        title_embeddings.append(find_field_vector(fellow_id, current_year, first_pub_year))

    multi_year_features = np.array(multi_year_features, dtype=np.float32)
    title_embeddings = np.array(title_embeddings, dtype=np.float32)
    multi_year_features = np.concatenate((multi_year_features, title_embeddings), axis=1)

    # fill zeros
    front_fill_year_count = first_pub_year - start_year
    front_filled_year_features = np.zeros((front_fill_year_count, multi_year_features.shape[1]),
                                          dtype=np.float32)

    back_fill_year_count = cut_year - extract_year
    back_filled_year_features = np.zeros((back_fill_year_count, multi_year_features.shape[1]),
                                         dtype=np.float32)

    return np.concatenate((front_filled_year_features, multi_year_features, back_filled_year_features))


def main():
    train_features = []
    test_features = []

    train_targets = []
    test_targets = []

    # test data
    test_fellow_ids = \
        single_type_fellow_data.loc[(single_type_fellow_data['type2'] == 'fellow')
                                    & (single_type_fellow_data['select_year'] > cut_year), 'id'].unique()
    test_fellow_ids = list(test_fellow_ids)
    for fellow_id in tqdm(test_fellow_ids):
        if fellow_id not in saved_fellow_ids:
            continue
        fellow_data = single_type_fellow_data.loc[single_type_fellow_data['id'] == fellow_id]
        select_year = fellow_data.iloc[0]['select_year']
        first_pub_year = fellow_data.iloc[0]['first_pub_year']

        one_fellow_features = fellow_data_2_np_array(fellow_data, first_pub_year, cut_year)
        test_features.append(one_fellow_features)
        test_targets.append(select_year - cut_year)

    # train data
    process_bar = tqdm(range(first_select_year, cut_year + 1))
    for current_year in process_bar:
        fellow_ids = \
            all_fellow_data.loc[(all_fellow_data['type1'] == fellow_type)
                                & (all_fellow_data['type2'] == 'fellow')
                                & (all_fellow_data['current_year'] == current_year)
                                & ((all_fellow_data['select_year'] == current_year)
                                   | (all_fellow_data['select_year'] > current_year)
                                   & (all_fellow_data['continue_year'] > 7)), 'id'].unique()
        fellow_ids = list(fellow_ids)
        process_bar.set_description("year:{} -> size:{}".format(current_year, len(fellow_ids)))

        for fellow_id in fellow_ids:
            if fellow_id in test_fellow_ids:
                continue
            fellow_data = single_type_fellow_data.loc[single_type_fellow_data['id'] == fellow_id]
            select_year = fellow_data.iloc[0]['select_year']
            first_pub_year = fellow_data.iloc[0]['first_pub_year']

            one_fellow_features = fellow_data_2_np_array(fellow_data, first_pub_year, current_year)
            train_features.append(one_fellow_features)
            train_targets.append(select_year - current_year)

    train_features = np.array(train_features, dtype=np.float32)
    test_features = np.array(test_features, dtype=np.float32)

    train_targets = np.array(train_targets, dtype=np.int32)
    test_targets = np.array(test_targets, dtype=np.int32)

    np.save(os.path.join(data_dir_path, 'train_features.npy'), train_features)
    np.save(os.path.join(data_dir_path, 'test_features.npy'), test_features)

    np.save(os.path.join(data_dir_path, 'train_targets.npy'), train_targets)
    np.save(os.path.join(data_dir_path, 'test_targets.npy'), test_targets)


if __name__ == '__main__':
    fellow_types = ['acm', 'ieee']
    for fellow_type in fellow_types:
        start_year = 1936
        first_select_year = 1952
        cut_year_range: list
        if 'acm' == fellow_type:
            start_year = 1936
            first_select_year = 1994
            cut_year_range = range(2009, 2019)
        elif 'ieee' == fellow_type:
            start_year = 1919
            first_select_year = 1952
            cut_year_range = range(2010, 2020)
        feature_cols = ['org_id', 'pub_num', 'pub_alpha', 'cite_num', 'cite_cum', 'cite_alpha', 'h_index',
                        'i10_index', 'co_distance_{}'.format(fellow_type), 'pub_per_year', 'cite_per_year',
                        'continue_year']

        all_fellow_data = pd.read_csv(os.path.join('..', 'data', 'dynamic_fellow_data.csv'.format(fellow_type)))
        single_type_fellow_data = \
            pd.read_csv(os.path.join('..', 'data', 'dynamic_fellow_data_{}.csv'.format(fellow_type)))

        title_embedding_dict = pickle.load(open(os.path.join('..', 'data', 'tech_area_{}.pkl'.format(fellow_type)), 'rb'))
        saved_fellow_ids = set(map(lambda k: k[0], title_embedding_dict.keys()))

        cut_year: int
        for year in cut_year_range:
            cut_year = year
            data_dir_path = Path(os.path.join('..', 'data', 'reg_data', fellow_type, str(cut_year)))
            data_dir_path.mkdir(parents=True, exist_ok=True)
            print('{}: cut year: {}'.format(fellow_type, cut_year))
            main()
