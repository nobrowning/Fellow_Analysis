import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold
from os import environ, path


class ModelSelector:
    def __init__(self, learning_rate=0.01, batch_size=128, epochs=10, times=5):
        self.times = times
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.kf = KFold(n_splits=5)

    # def get_data_path(self) -> str:
    #     data_path = environ.get('FELLOW_DATA_PATH')
    #     if data_path is None:
    #         raise Exception('"FELLOW_DATA_PATH" not config')
    #     return path.expanduser(data_path)

    def load_cls_data(self, cut_year: int, fellow_type: str):
        # data_path = self.get_data_path()
        data_path = path.join('..', 'data')
        features = np.load(path.join(data_path, 'cls_data', fellow_type, str(cut_year), 'all_features.npy'))
        targets = np.load(path.join(data_path, 'cls_data', fellow_type, str(cut_year), 'all_targets.npy'))

        # target one hot
        targets = tf.one_hot(targets.reshape([-1, 1]), 2)
        targets = tf.reshape(targets, [-1, 2]).numpy()

        # data size
        sample_num = targets.shape[0]
        step_num = features.shape[1]
        feature_dim = features.shape[2]
        return features, targets, sample_num, step_num, feature_dim

    def split_cls_data(self, features, targets, train_idx, val_idx):
        trainX = features[train_idx]
        trainT = targets[train_idx]
        testX = features[val_idx]
        testT = targets[val_idx]

        trainX = tf.dtypes.cast(trainX, tf.float32)
        testX = tf.dtypes.cast(testX, tf.float32)
        trainT = tf.dtypes.cast(trainT, tf.float32)
        testT = tf.dtypes.cast(testT, tf.float32)

        train_dataset = tf.data.Dataset.from_tensor_slices((trainX, trainT))
        train_dataset = train_dataset.batch(self.batch_size)

        test_dataset = tf.data.Dataset.from_tensor_slices((testX, testT))
        test_dataset = test_dataset.batch(self.batch_size)

        return train_dataset, test_dataset

    def load_reg_data(self, cut_year: int, fellow_type: str):
        # data_path = self.get_data_path()
        data_path = path.join('..', 'data')
        train_features = np.load(path.join(data_path, 'rnn_data_year', fellow_type, str(cut_year), 'train_features.npy'))
        test_features = np.load(path.join(data_path, 'rnn_data_year', fellow_type, str(cut_year), 'test_features.npy'))

        train_targets = np.load(path.join(data_path, 'rnn_data_year', fellow_type, str(cut_year), 'train_targets.npy'))
        test_targets = np.load(path.join(data_path, 'rnn_data_year', fellow_type, str(cut_year), 'test_targets.npy'))

        testT = test_targets.reshape([-1, 1])
        trainT = train_targets.reshape([-1, 1])
        trainX = train_features
        testX = test_features

        sample_num = train_targets.shape[0] + test_targets.shape[0]
        step_num = train_features.shape[1]
        feature_dim = train_features.shape[2]

        trainX = tf.dtypes.cast(trainX, tf.float32)
        testX = tf.dtypes.cast(testX, tf.float32)
        trainT = tf.dtypes.cast(trainT, tf.float32)
        testT = tf.dtypes.cast(testT, tf.float32)

        train_dataset = tf.data.Dataset.from_tensor_slices((trainX, trainT))
        train_dataset = train_dataset.shuffle(1000).batch(self.batch_size)

        test_dataset = tf.data.Dataset.from_tensor_slices((testX, testT))
        test_dataset = test_dataset.shuffle(1000).batch(self.batch_size)

        return train_dataset, test_dataset, sample_num, step_num, feature_dim

    def run_cls_model(self, cut_year: int, fellow_type: str, model_class, add_param: bool, model_parameters: dict):
        print("******************** {}-{} ********************".format(fellow_type, cut_year))
        # load data
        features, targets, sample_num, step_num, feature_dim = self.load_cls_data(cut_year, fellow_type)
        if add_param:
            model_parameters['max_seq_length'] = step_num
            model_parameters['d_model'] = feature_dim + 3
            model_parameters['max_seq_length'] = step_num

        s_results = []
        for train_idx, val_idx in self.kf.split(features, targets):

            train_dataset, test_dataset = self.split_cls_data(features, targets, train_idx, val_idx)

            results = []
            for t in range(self.times):
                model = model_class(**model_parameters)
                bce = tf.keras.losses.BinaryCrossentropy()
                train_loss = tf.keras.metrics.Mean(name='train_loss')
                test_loss = tf.keras.metrics.Mean(name='test_loss')
                test_acc = tf.keras.metrics.Accuracy(name='test_acc')
                optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

                test_precision = tf.keras.metrics.Precision(name='test_precision')
                test_recall = tf.keras.metrics.Recall(name='test_recall')

                train_step_signature = [
                    tf.TensorSpec(shape=(None, step_num, feature_dim), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
                ]
                @tf.function(input_signature=train_step_signature)
                def train_step(inp, tar):

                    with tf.GradientTape() as tape:
                        logits = model(inp, True)
                        loss = bce(tar, logits)

                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                    train_loss(loss)

                test_step_signature = [
                    tf.TensorSpec(shape=(None, step_num, feature_dim), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
                ]
                @tf.function(input_signature=test_step_signature)
                def test_step(inp, tar):
                    logits = model(inp, False)
                    loss = bce(tar, logits)
                    test_loss(loss)

                    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                    labels = tf.argmax(tar, axis=-1, output_type=tf.int32)

                    test_acc.update_state(labels, predictions)
                    test_precision.update_state(labels, predictions)
                    test_recall.update_state(labels, predictions)

                best_epoch_result = 0
                for epoch in range(self.epochs):
                    train_loss.reset_states()
                    test_loss.reset_states()
                    test_acc.reset_states()
                    test_precision.reset_states()
                    test_recall.reset_states()

                    for (batch, (inp, tar)) in enumerate(train_dataset):
                        train_step(inp, tar)

                    for (batch, (inp, tar)) in enumerate(test_dataset):
                        test_step(inp, tar)

                    precision = test_precision.result()
                    recall = test_recall.result()
                    F1 = 2 * (precision * recall) / (precision + recall) * 100

                    if F1 > best_epoch_result:
                        best_epoch_result = F1

                results.append(best_epoch_result)
            results.sort()
            results.remove(results[0])
            results.remove(results[1])
            s_result = sum(results) / len(results)
            s_results.append(s_result)
            print('F1:{:.1f}'.format(s_result))
        print('--------------------F1 avg: {:.1f}--------------------'.format(sum(s_results) / len(s_results)))

    def mae(self, target, predict):
        loss = target - predict
        loss = tf.square(loss)
        loss = tf.sqrt(loss)
        return tf.reduce_mean(loss)

    def run_reg_model(self, cut_year: int, fellow_type: str, model_class, add_param: bool, model_parameters):
        print("******************** {}-{} ********************".format(fellow_type, cut_year))
        train_dataset, test_dataset, sample_num, step_num, feature_dim = self.load_reg_data(cut_year, fellow_type)
        if add_param:
            model_parameters['max_seq_length'] = step_num
            model_parameters['d_model'] = feature_dim + 3
            model_parameters['max_seq_length'] = step_num

        results = []
        for t in range(self.times):
            model = model_class(**model_parameters)
            train_loss = tf.keras.metrics.Mean(name='train_loss')
            test_loss = tf.keras.metrics.Mean(name='test_loss')
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

            train_step_signature = [
                tf.TensorSpec(shape=(None, step_num, feature_dim), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            ]
            @tf.function(input_signature=train_step_signature)
            def train_step(inp, tar):

                with tf.GradientTape() as tape:
                    predictions = model(inp, True)
                    loss = self.mae(tar, predictions)

                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                train_loss(loss)

            train_step_signature = [
                tf.TensorSpec(shape=(None, step_num, feature_dim), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            ]
            @tf.function(input_signature=train_step_signature)
            def test_step(inp, tar):

                predictions = model(inp, False)
                loss = self.mae(tar, predictions)

                test_loss(loss)

            best_epoch_result = 999
            for epoch in range(self.epochs):
                train_loss.reset_states()
                test_loss.reset_states()

                for (batch, (inp, tar)) in enumerate(train_dataset):
                    train_step(inp, tar)

                for (batch, (inp, tar)) in enumerate(test_dataset):
                    test_step(inp, tar)

                if test_loss.result() < best_epoch_result:
                    best_epoch_result = test_loss.result()

            results.append(best_epoch_result)
            print('time-{} loss:{:.4f}'.format(t, best_epoch_result))
        print('--------------------lost avg: {:.4f}--------------------'.format(sum(results) / len(results)))
