import tensorflow as tf
from data_loader import *
from abc import abstractmethod
from preprocessing import Preprocessing
import scipy.io.wavfile as wavfile

class BaseNN:
    def __init__(self, train_features_dir, val_features_dir, test_audios_dir,  num_epochs, train_batch_size,
                 val_batch_size, learning_rate, base_dir, max_to_keep, model_name): #test_features_dir, test_batch_size

        self.data_loader = DataLoader(train_features_dir, val_features_dir,
                                      train_batch_size, val_batch_size)
        self.test_audios_dir = test_audios_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.base_dir = base_dir
        self.model_name= model_name
        self.max_to_keep = max_to_keep
        self.checkpoint_dir = os.path.join(self.base_dir, self.model_name, "checkpoints")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.summary_dir = os.path.join(self.base_dir, self.model_name, "summaries")
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        prep = Preprocessing()
        self.sample_rate = prep.sample_rate
        self.frame_dim = int(prep.frame_len/2)+1
        self.frame_count = prep.frame_count
        self.preprocess_test_data = prep.preprocess_test_data
        self.preprocess_test_data_unet = prep.preprocess_test_data_unet
        self.produce_time_outputs = prep.produce_time_outputs
        self.time_outputs_into_track = prep.time_outputs_into_track
        self.produce_time_outputs_unet = prep.produce_time_outputs_unet
        self.time_outputs_into_track_unet = prep.time_outputs_into_track_unet

    def create_network(self):
        self.X = tf.placeholder(tf.float32, [None, self.frame_count, self.frame_dim], name="X")
        self.y = tf.placeholder(tf.float32, [None, self.frame_count, self.frame_dim], name='y')
        self.prediction = self.network(self.X)
        self.los = self.loss(self.y, self.prediction)
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.optimiser = self.define_optim()
        pass


    def define_optim(self):
        return tf.train.AdamOptimizer(self.learning_rate).minimize(self.los, global_step=self.global_step)

    def initialize_network(self):
        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)
        checkpoint = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if checkpoint:
            checkpoint_path = checkpoint.model_checkpoint_path
            print("Restoring from checkpoint!")
            self.saver.restore(self.sess, checkpoint_path)
        else:
            self.sess.run(tf.global_variables_initializer())

        self.summary_op = tf.summary.merge_all()
        self.train_summary_writer = tf.summary.FileWriter(os.path.join(self.summary_dir, "train"), self.sess.graph)
        self.val_summary_writer = tf.summary.FileWriter(os.path.join(self.summary_dir, "validation"), self.sess.graph)

        s = 0
        # for v in tf.trainable_variables():
        #     sh = v.shape
        #     s += sh[0] * sh[1] * sh[2] * sh[3]
        # print(s)
        # exit()


    def train_model(self, display_step, validation_step, checkpoint_step, summary_step):
        validation_index = 0
        num_batches = np.ceil(80000/self.train_batch_size)
        num_val_batches = np.ceil(19293/self.val_batch_size)
        for epoch in range(self.num_epochs):
            try:
                np.random.shuffle(self.data_loader.train_paths)
                np.random.shuffle(self.data_loader.val_paths)
                for i in range(int(num_batches)):
                    batch_x, batch_y = self.data_loader.train_data_loader(i)
                    _, loss_value, summary_str, global_step = self.sess.run([self.optimiser, self.los, self.summary_op, self.global_step], feed_dict={self.X: batch_x, self.y: batch_y})
                    if global_step % display_step == 0: #display and summary
                        print("epoch number: ", epoch)
                        print("For iter: ", global_step)
                        print("Train Loss: ", loss_value)
                        print("__________________")
                    if global_step % summary_step == 0:
                        self.train_summary_writer.add_summary(summary_str, global_step)
                    if global_step % validation_step == 0:
                        batch_x, batch_y = self.data_loader.val_data_loader(validation_index)
                        validation_los =  self.sess.run(self.los, feed_dict={self.X: batch_x, self.y: batch_y})
                        if validation_index < num_val_batches:
                            validation_index += 1
                        else:
                            validation_index = 0
                        print("Validation loss: ", validation_los)
                        if global_step % summary_step == 0:
                            self.val_summary_writer.add_summary(summary_str, global_step)

                    if global_step % checkpoint_step == 0:
                        self.saver.save(self.sess, os.path.join(self.checkpoint_dir, self.model_name + ".ckpt"), global_step=global_step)
            except Exception as e:
                print(e)
                print("something happened")


    def test_model(self):
        print("testing..")
        mix_dir = self.test_audios_dir
        track_paths = glob.glob(mix_dir + "*.wav")
        for track_path in track_paths:
            inputs = self.preprocess_test_data(track_path) #use preprocess_test_data if it is not unet
            ####################
            # labels = self.preprocess_test_data(track_path.replace("mix_tracks", "labels"))
            # inputs = np.array(inputs)
            # labels = np.array(labels)
            # ideal_rm = np.sqrt(labels ** 2 / (labels ** 2 + inputs ** 2 + 1e-7))
            # piano_ft = ideal_rm * inputs
            ####################
            #make input dimensions 332,241
            Y_pred = self.sess.run(self.prediction, feed_dict={self.X: inputs})
            print("Y_pred: ",Y_pred)
            print("inputs: ", inputs)
            rm = (Y_pred)**2
            piano_ft = rm * inputs
            time_outputs = self.produce_time_outputs(piano_ft)
            track = self.time_outputs_into_track(time_outputs, 30 * self.sample_rate)
            wavfile.write(track_path.replace("mix", "cleaned"), 16000, track.astype("int16"))
        print("tested")

    def test_model_unet(self):
        print("testing..")
        mix_dir = self.test_audios_dir
        track_paths = glob.glob(mix_dir + "*.wav")
        for track_path in track_paths:
            inputs = self.preprocess_test_data_unet(track_path)  # tvec 320 241 inputner
            ####################
            # labels = self.preprocess_test_data(track_path.replace("mix_tracks", "labels"))
            # inputs = np.array(inputs)
            # labels = np.array(labels)
            # ideal_rm = np.sqrt(labels ** 2 / (labels ** 2 + inputs ** 2 + 1e-7))
            # piano_ft = ideal_rm * inputs
            ####################
            # make input dimensions 332,241
            print("inputs: ", inputs.shape)
            npad = ((0, 0), (12, 0), (0, 0))
            inputs_padded = np.pad(inputs, pad_width=npad, mode='constant', constant_values=0)
            print("inputs padded: ", inputs_padded.shape)
            Y_pred = self.sess.run(self.prediction, feed_dict={self.X: inputs_padded})
            print("Y_pred: ", Y_pred.shape)

            npad = ((0, 0), (0, 0), (1, 0))
            Y_pred = np.pad(Y_pred, pad_width=npad, mode='constant', constant_values=0)
            print("Y_pred padded: ", Y_pred.shape)
            rm = (Y_pred) ** 2
            piano_ft = rm * inputs
            time_outputs = self.produce_time_outputs_unet(piano_ft, input_sec = 4.815)
            print("time_outputs.shape: ",time_outputs.shape)
            track = self.time_outputs_into_track_unet(time_outputs, 30 * self.sample_rate, input_sec = 4.815)
            wavfile.write(track_path.replace("mix", "cleaned"), 16000, track.astype("int16"))
        print("tested")

    @abstractmethod
    def network(self, X):
        raise NotImplementedError('subclasses must override network()!')

    @abstractmethod
    def loss(self, Y, Y_pred):
        raise NotImplementedError('subclasses must override loss()!')
