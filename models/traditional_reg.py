import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import os


def baseline(model, trainX, trainT, testX, testT, parameters={}):
    if len(parameters) == 0:
        amodel = model()
    else:
        amodel = model(**parameters)
    amodel.fit(trainX, trainT)
    predy = abs(amodel.predict(testX))
    error = np.mean(np.sqrt(np.square(predy - testT))).round(2)
    return error, predy, amodel


if __name__ == '__main__':
    fellow_types = ['acm', 'ieee']
    algorithms = {
        RandomForestRegressor: {'n_estimators': 10, 'min_samples_leaf': 32, 'max_depth': 3},
        DecisionTreeRegressor: {'min_samples_leaf': 32, 'max_depth': 3},
        LinearRegression: {},
        Lasso: {'alpha': 1e-1},
        Ridge: {'alpha': 10e-3},
        ElasticNet: {'alpha': 10e-1},
        SVR: {'C': 1},  # 'C': 1
        MLPRegressor: dict(hidden_layer_sizes=(64, 32), learning_rate='adaptive',
                           max_iter=2000, learning_rate_init=0.005, activation='logistic',
                           alpha=10e-5, solver='adam')
    }

    data_root_path = os.environ.get('FELLOW_DATA_PATH')
    if data_root_path is None:
        raise Exception('"FELLOW_DATA_PATH" not config')
    data_root_path = os.path.expanduser(data_root_path)

    for fellow_type in fellow_types:

        cut_year_range = range(2015, 2020)
        if 'acm' == fellow_type:
            cut_year_range = range(2009, 2019)
        elif 'ieee' == fellow_type:
            cut_year_range = range(2010, 2020)

        for cut_year in cut_year_range:
            print("******************** {}-{} ********************".format(fellow_type, cut_year))
            trainX = np.load(os.path.join(data_root_path, 'rnn_data_year/{}/{}/train_features.npy'.format(fellow_type, cut_year)))
            trainT = np.load(os.path.join(data_root_path, 'rnn_data_year/{}/{}/train_targets.npy'.format(fellow_type, cut_year)))
            testX = np.load(os.path.join(data_root_path, 'rnn_data_year/{}/{}/test_features.npy'.format(fellow_type, cut_year)))
            testT = np.load(os.path.join(data_root_path, 'rnn_data_year/{}/{}/test_targets.npy'.format(fellow_type, cut_year)))

            trainX = trainX.reshape([trainX.shape[0], -1])
            testX = testX.reshape([testX.shape[0], -1])

            for alg, param in algorithms.items():
                print(alg.__name__, end=':\t')
                er, py, m = baseline(alg, trainX, trainT, testX, testT, param)
                print(er)
