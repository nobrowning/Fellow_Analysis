import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import warnings
import os


def baseline(model, trainX, trainT, testX, testT, parameters={}):
    if len(parameters) == 0:
        amodel = model()
    else:
        amodel = model(**parameters)
    amodel.fit(trainX, trainT)
    pred_y = abs(amodel.predict(testX))
    accuracy = round(f1_score(testT, pred_y), 4)
    return accuracy, pred_y, amodel


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    fellow_types = ['acm', 'ieee']
    algorithms = {
        LogisticRegression: {},
        RidgeClassifier: {},
        RandomForestClassifier: {'n_estimators': 32, 'min_samples_leaf': 32, 'max_depth': 3},
        KNeighborsClassifier: {'n_neighbors': 2},
        GaussianNB: {},
        DecisionTreeClassifier: {'min_samples_leaf': 32, 'max_depth': 3},
        SVC: {'C': 1},
        MLPClassifier: dict(hidden_layer_sizes=(64, 32), learning_rate='adaptive',
                            max_iter=2000, learning_rate_init=0.005, activation='logistic',
                            alpha=10e-5, solver='adam')
    }

    for fellow_type in fellow_types:

        cut_year_range = range(2015, 2020)
        if 'acm' == fellow_type:
            cut_year_range = range(2015, 2019 + 1)
        elif 'ieee' == fellow_type:
            cut_year_range = range(2016, 2020 + 1)
        
        data_root_path = os.environ.get('FELLOW_DATA_PATH')
        if data_root_path is None:
            raise Exception('"FELLOW_DATA_PATH" not config')
        data_root_path = os.path.expanduser(data_root_path)

        for cut_year in cut_year_range:
            print("******************** {}-{} ********************".format(fellow_type, cut_year))

            # load data
            features = np.load(os.path.join(data_root_path, "cls_data/{}/{}/all_features.npy".format(fellow_type, cut_year)))
            targets = np.load(os.path.join(data_root_path, "cls_data/{}/{}/all_targets.npy".format(fellow_type, cut_year)))
            features = features.reshape([features.shape[0], -1])

            for alg, param in algorithms.items():
                print(alg.__name__, end=':\t')

                results = []
                kf = KFold(n_splits=5)
                for train_idx, val_idx in kf.split(features, targets):
                    trainX = features[train_idx]
                    trainT = targets[train_idx]
                    testX = features[val_idx]
                    testT = targets[val_idx]
                    acc, py, m = baseline(alg, trainX, trainT, testX, testT, param)
                    results.append(acc)
                print(round(sum(results)/len(results)*100, 1))

