import tensorflow as tf
from model_selector import ModelSelector


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def __call__(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)

        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Model(tf.keras.Model):
    def __init__(self, rnn_cell=tf.keras.layers.GRUCell, cell_units=[64, 32, 32, 32]):
        super(Model, self).__init__(name='')

        # org id embedding
        self.embedding = tf.keras.layers.Embedding(input_dim=1281, output_dim=4)  # , kernel_initializer='ones'

        rnn_layers = [rnn_cell(size) for size in cell_units]
        multi_rnn_cell = tf.keras.layers.StackedRNNCells(rnn_layers)
        self.multi_rnn_layer = tf.keras.layers.RNN(multi_rnn_cell, return_state=True, return_sequences=True)
        self.attention = BahdanauAttention(cell_units[-1])
        self.dense = tf.keras.layers.Dense(2, activation=tf.keras.activations.softmax)

    def __call__(self, input_tensor, training=False):
        # split feature
        other_features0, org_ids, other_features = tf.split(input_tensor, [1, 1, input_tensor.shape[2] - 2], axis=2)

        # org id embedding
        org_id_emb = self.embedding(org_ids)[:, :, -1, :]

        inputs = tf.concat([other_features0, other_features, org_id_emb], axis=2)

        outputs, rnn_status_0, rnn_status_1, rnn_status_2, rnn_status_3 = self.multi_rnn_layer(inputs)
        context_vector, attention_weights = self.attention(rnn_status_3, outputs)
        outputs = self.dense(context_vector)
        return outputs


if __name__ == '__main__':
    fellow_types = ['acm', 'ieee']
    cut_year_range_dict = {
      'ieee': range(2016, 2020 + 1),
      'acm': range(2015, 2019 + 1)
    }

    model_selector = ModelSelector(learning_rate=0.001, batch_size=128, epochs=10, times=5)

    for f_type in fellow_types:
        for c_y in cut_year_range_dict[f_type]:
            model_selector.run_cls_model(c_y, f_type, Model, False, {})
