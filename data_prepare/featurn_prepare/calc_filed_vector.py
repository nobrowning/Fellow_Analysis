import pymysql
import requests
import numpy as np
from tqdm import tqdm
from typing import List
import pickle


def bert_encode(sentences: List[str]) -> np.array:
    data = {"id": 123, "texts": sentences, "is_tokenized": False}
    response = requests.post(encode_url, json=data)
    if response.status_code == 200:
        r_body = response.json()
        embedding_vectors = np.array(r_body.get("result"))
        # return embedding_vectors.mean(axis=0)
        return embedding_vectors
    return None


def main():
    db = pymysql.connect("127.0.0.1", "root", "root", "fellow_20_mag",
                         charset='utf8', cursorclass=pymysql.cursors.SSDictCursor)
    cursor = db.cursor()
    cursor.execute("SELECT id, MIN(current_year) as first_pub_year, MAX(current_year) as last_pub_year "
                   "FROM dynamic_full_fellow_feature "
                   "GROUP BY id")

    fellows = cursor.fetchall()
    fellow_per_year_vec = dict()

    process_bar = tqdm(fellows)
    for fellow in process_bar:
        fellow_id = fellow['id']
        first_pub_year = fellow['first_pub_year']
        last_pub_year = fellow['last_pub_year']
        title_embedding = None

        for year in range(first_pub_year, last_pub_year + 1):
            process_bar.set_description("id:{} year:{}/{}".format(fellow_id, year, last_pub_year))
            cursor.execute("SELECT title, abstract FROM p_paper "
                           "WHERE id IN ( "
                           "SELECT paper_id FROM p_paper_fellow_relation "
                           "WHERE fellow_id=%s AND year=%s)", (fellow_id, year))
            titles = list(map(lambda p: p['title'] + '||' + p['abstract'], cursor.fetchall()))

            if len(titles) == 0:
                pass
            else:
                one_year_title_embedding = bert_encode(titles)
                if one_year_title_embedding is not None:

                    if title_embedding is None:
                        title_embedding = one_year_title_embedding
                    else:
                        title_embedding = np.concatenate([title_embedding, one_year_title_embedding])

            fellow_per_year_vec[(fellow_id, year)] = title_embedding.mean(axis=0)

    with open('tech_area_{}.pkl'.format(field_system_type), 'wb') as f:
        pickle.dump(fellow_per_year_vec, f)


if __name__ == '__main__':
    field_system_type = 'acm' # 'ieee'
    encode_url = 'http://10.2.2.9:5933/encode'
    main()

