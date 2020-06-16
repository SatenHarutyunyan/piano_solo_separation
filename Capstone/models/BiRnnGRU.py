from .BaseNN import *


class BiRnnGRU(BaseNN):

    def network(self, batch_X):
        seq_len = self.frame_count  # time_steps
        num_units = 1000  # hyperameter
        input = tf.unstack(batch_X, seq_len, 1)
        gru_layer_fw = tf.contrib.rnn.GRUCell(num_units)
        gru_layer_bw = tf.contrib.rnn.GRUCell(num_units)
        initial_state_fw = self._rnn_placeholders(gru_layer_fw.zero_state(tf.shape(input)[0], tf.float32),
                                                  "c_state_fw")
        initial_state_bw = self._rnn_placeholders(gru_layer_bw.zero_state(tf.shape(input)[0], tf.float32),
                                                  "c_state_bw")
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(gru_layer_fw, gru_layer_bw, input,
                                                     sequence_length=seq_len,
                                                     initial_state_fw=initial_state_fw,
                                                     initial_state_bw=initial_state_bw, dtype="float32")
        outputs = tf.stack(outputs, axis=1)
        # outputs = tf.concat(outputs, 2)
        dense1 = tf.layers.dense(inputs=outputs, units=3000, activation=tf.nn.tanh, name="dense1")
        lin1 = tf.layers.dense(inputs=outputs, units=3000, activation=None, name="lin1")
        sum1 = dense1 + lin1
        dense2 = tf.layers.dense(inputs=sum1, units=3000, activation=tf.nn.relu, name="dense2")
        lin2 = tf.layers.dense(inputs=sum1, units=3000, activation=None, name="lin2")
        sum2 = dense2 + lin2
        prediction = tf.layers.dense(outputs, self.frame_dim, activation=tf.nn.sigmoid, name="last_dense")
        return prediction

    def loss(self, y, y_pred):
        l2_loss = tf.reduce_mean(tf.square(y - y_pred))
        tf.summary.scalar("Loss", l2_loss)
        return l2_loss

    def _rnn_placeholders(self, state, c_name):
        c = state
        c = tf.placeholder_with_default(c, c.shape, c_name)
        # h = tf.placeholder_with_default(h, h.shape, h_name)
        # return tf.contrib.rnn.LSTMStateTuple(c, h)
        return c

