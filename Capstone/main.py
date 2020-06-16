import tensorflow as tf
# from models.BiRnnGRU import *
# from models.RNN import *
from models.UNET import *

# Datasets
tf.app.flags.DEFINE_string('train_features_dir', '/home/student/Saten/data/train/inputs/', 'Training features data directory.')
tf.app.flags.DEFINE_string('val_features_dir', '/home/student/Saten/data/validation/inputs/', 'Validation features data directory.')
tf.app.flags.DEFINE_string('test_audios_dir', '/home/student/Saten/data/test/mix_tracks/', 'Testing audios data directory.')

tf.app.flags.DEFINE_boolean('train', True, 'whether to train the network')
tf.app.flags.DEFINE_integer('num_epochs', 100000, 'epochs to train')#10000
tf.app.flags.DEFINE_integer('train_batch_size', 15, 'number of elements in a training batch')
tf.app.flags.DEFINE_integer('val_batch_size', 15, 'number of elements in a validation batch')

tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate of the optimizer')

tf.app.flags.DEFINE_integer('display_step', 20, 'Number of steps we cycle through before displaying detailed progress.')
tf.app.flags.DEFINE_integer('validation_step', 60, 'Number of steps we cycle through before validating the model.')#make it multiple of summary step

tf.app.flags.DEFINE_string('base_dir', './results', 'Directory in which results will be stored.')
tf.app.flags.DEFINE_integer('checkpoint_step', 300, 'Number of steps we cycle through before saving checkpoint.')#make it multiple of validation step
tf.app.flags.DEFINE_integer('max_to_keep', 5, 'Number of checkpoint files to keep.')

tf.app.flags.DEFINE_integer('summary_step', 20, 'Number of steps we cycle through before saving summary.')

tf.app.flags.DEFINE_string('model_name', 'UNET', 'name of model')

FLAGS = tf.app.flags.FLAGS


def main(argv=None):
    model = UNET(
        train_features_dir=FLAGS.train_features_dir,
        val_features_dir=FLAGS.val_features_dir,
        test_audios_dir = FLAGS.test_audios_dir,
        num_epochs=FLAGS.num_epochs,
        train_batch_size=FLAGS.train_batch_size,
        val_batch_size=FLAGS.val_batch_size,
        learning_rate=FLAGS.learning_rate,
        base_dir=FLAGS.base_dir,
        max_to_keep=FLAGS.max_to_keep,
        model_name=FLAGS.model_name,
    )

    model.create_network()
    model.initialize_network()

    if FLAGS.train:
        model.train_model(FLAGS.display_step, FLAGS.validation_step, FLAGS.checkpoint_step, FLAGS.summary_step)
    else:
        model.test_model_unet() #model.test_model() for other model


if __name__ == "__main__":
    tf.app.run()
