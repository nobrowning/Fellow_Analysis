import pandas as pd
import numpy as np
from tqdm import tqdm
import os


def main():
    single_type_fellow_data = f_data.loc[(f_data['type1'] == fellow_type) | (f_data['type2'] != 'fellow')]
    single_type_fellow_data = single_type_fellow_data.fillna(0)

    zero_0_scaler = lambda x: (x - np.mean(x)) / (np.std(x))
    for col_name in tqdm(feature_cols):
        one_col_data = single_type_fellow_data[[col_name]].apply(zero_0_scaler)
        single_type_fellow_data = single_type_fellow_data.drop([col_name], axis=1)
        single_type_fellow_data = pd.concat([single_type_fellow_data, one_col_data], axis=1)
    single_type_fellow_data = single_type_fellow_data.set_index(['id', 'current_year'])
    single_type_fellow_data.to_csv(os.path.join(data_path, 'dynamic_fellow_data_{}.csv'.format(fellow_type)))
    pass


if __name__ == '__main__':
    fellow_types = ['acm', 'ieee']
    data_path = os.path.join('..', 'data')
    f_data = pd.read_csv(os.path.join(data_path, 'dynamic_fellow_data.csv'))

    for fellow_type in fellow_types:
        feature_cols = ['pub_num', 'pub_alpha', 'cite_num', 'cite_cum', 'cite_alpha', 'h_index',
                        'i10_index', 'co_distance_{}'.format(fellow_type), 'pub_per_year', 'cite_per_year',
                        'continue_year']
        main()
