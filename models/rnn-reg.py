import tensorflow as tf
from model_selector import ModelSelector


class Model(tf.keras.Model):
    def __init__(self, rnn_cell=tf.keras.layers.GRUCell, cell_units=[64, 32, 32, 32]):
        super(Model, self).__init__()
        
        # org id embedding
        self.embedding = tf.keras.layers.Embedding(input_dim=1281, output_dim=4) # , kernel_initializer='ones'
        
        # rnn
        rnn_layers = [rnn_cell(size) for size in cell_units]
        multi_rnn_cell = tf.keras.layers.StackedRNNCells(rnn_layers)
        self.multi_rnn_layer = tf.keras.layers.RNN(multi_rnn_cell, return_state=False, return_sequences=True)
        
        # regression result
        self.dense = tf.keras.layers.Dense(1, kernel_initializer='ones') # , kernel_initializer='ones'
    
    def call(self, input_tensor, training=False):
        # split feature
        other_features0, org_ids, other_features = tf.split(input_tensor, [1, 1, input_tensor.shape[2] - 2], axis=2) 
        
        # org id embedding
        org_id_emb = self.embedding(org_ids)[:, :, -1, :]
        
        # desc word embedding
        inputs = tf.concat([other_features0, other_features, org_id_emb], axis=2)
        outputs = self.multi_rnn_layer(inputs)
        outputs = self.dense(outputs[:, -1, :])
        return outputs


if __name__ == '__main__':
    fellow_types = ['acm', 'ieee']
    cut_year_range_dict = {
      'ieee': range(2010, 2020),
      'acm': range(2009, 2019)
    }

    model_selector = ModelSelector(learning_rate=0.001, batch_size=128, epochs=10, times=5)

    for f_type in fellow_types:
        for c_y in cut_year_range_dict[f_type]:
            model_selector.run_reg_model(c_y, f_type, Model, False, {})






