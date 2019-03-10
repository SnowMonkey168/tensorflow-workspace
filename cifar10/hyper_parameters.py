import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('version', 'rc_20181110', '''A version number defining the directory to save logs and checkpoints.''')
tf.app.flags.DEFINE_integer('report_freq', 100, '''Steps takes to output errors on the screen and write summaries.''')
tf.app.flags.DEFINE_float('train_ema_decay', 0.95, '''The decay factor of the train error's moving average shown on TensorBoard.''')
tf.app.flags.DEFINE_integer('train_steps', 80000, '''Total steps that you want to train.''')
tf.app.flags.DEFINE_boolean('is_full_validation', False, '''Validation w/ full validation set or a random batch.''')
tf.app.flags.DEFINE_integer('train_batch_size', 128, '''Train batch size.''')
tf.app.flags.DEFINE_integer('validation_batch_size', 250, '''Validation batch size, better to be a divisor of 10000 for this task.''')
tf.app.flags.DEFINE_integer('test_batch_size', 125, '''Test batch size.''')
tf.app.flags.DEFINE_float('init_learning_rate', 0.1, '''Initial learning rate.''')
tf.app.flags.DEFINE_float('lr_decay_factor', 0.1, '''How much to decay the learning rate each time.''')
tf.app.flags.DEFINE_integer('decay_step0', 40000, '''At which step to decay the weight learning rate.''')
tf.app.flags.DEFINE_integer('decay_step1', 60000, '''At which step to decay the bias learning rate.''')
tf.app.flags.DEFINE_integer('num_residual_blocks', 5, '''How many residual blocks do you want.''')
tf.app.flags.DEFINE_float('weight_decay', 0.0002, '''Scale for l2 regularization.''')
tf.app.flags.DEFINE_integer('padding_size', 2, '''In data augmentation, layers of zero padding on each side of the image.''')
tf.app.flags.DEFINE_string('ckpt_path', 'cache/logs_repeat20/model.ckpt-100000', '''Checkpoint directory of restore.''')
tf.app.flags.DEFINE_boolean('is_use_ckpt', False, '''Whether to load a checkpoint and continue training.''')
tf.app.flags.DEFINE_string('test_ckpt_path', 'model_110.ckpt-79999', '''Checkpoint directory to restore.''')

train_dir = 'logs_' + FLAGS.version + '/'
test_dir = 'testdata/'