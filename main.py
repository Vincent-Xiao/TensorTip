import tensorflow as tf
import importlib
import tensorflow.python.platform
import os
import numpy as np
from progress.bar import Bar
from datetime import datetime
from tensorflow.python.platform import gfile
from data import *
from evaluate import evaluate
import time

from ImageNetReading import image_processing

MOVING_AVERAGE_DECAY = 0.997
FLAGS = tf.app.flags.FLAGS
# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 256,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_epochs', 128,
                            """Number of epochs to train. -1 for unlimited""")
tf.app.flags.DEFINE_float('learning_rate', 1e-3,
                            """Initial learning rate used.""")
tf.app.flags.DEFINE_string('model', 'model',
                           """Name of loaded model.""")
tf.app.flags.DEFINE_string('save', None,
                           """Name of saved dir.""")
tf.app.flags.DEFINE_string('load', None,
                           """Name of loaded dir for resume training.""")
tf.app.flags.DEFINE_string('resume', False,
                           """if resume training or not.""")
tf.app.flags.DEFINE_string('dataset', 'cifar10',
                           """Name of dataset used.""")
tf.app.flags.DEFINE_string('gpu', True,
                           """use gpu.""")
tf.app.flags.DEFINE_string('device', 0,
                           """which gpu to use.""")
tf.app.flags.DEFINE_string('summary', True,
                           """Record summary.""")
tf.app.flags.DEFINE_string('log', 'INFO',
                           'The threshold for what messages will be logged '
                            """DEBUG, INFO, WARN, ERROR, or FATAL.""")
tf.app.flags.DEFINE_string('display_interval', None,
                           """Interval steps for displaying and summary train loss""")
tf.app.flags.DEFINE_string('test_interval', None,
                           """Interval steps for test loss and accuracy""")
tf.app.flags.DEFINE_integer('decay_steps', 1000,
                           """decay  steps for learning""")
tf.app.flags.DEFINE_float('decay_rate', 0.96,
                           """decay rate for learning""")
tf.app.flags.DEFINE_integer('num_threads', 8,
                           """num_threads for data processing""")
tf.app.flags.DEFINE_string('using_learning_rate_decay_fn', True,
                           """whether using learning_rate_decay_fn or not;if not , using optimtizer auto deacy  """)

currentTime=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
FLAGS.checkpoint_dir = FLAGS.load if FLAGS.resume else  './results/' + FLAGS.save+'/'+currentTime
FLAGS.log_dir = FLAGS.checkpoint_dir + '/log/'
FLAGS.loggingFile=FLAGS.checkpoint_dir+'.log'
#tf.logging.set_verbosity(FLAGS.log)

def count_params(var_list):
    num = 0
    for var in var_list:
        if var is not None:
            num += var.get_shape().num_elements()
    return num


def add_summaries(scalar_list=[], activation_list=[], var_list=[], grad_list=[]):

    for var in scalar_list:
        if var is not None:
            tf.summary.scalar(var.op.name, var)

    for grad, var in grad_list:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)
    for var in var_list:
        if var is not None:
            tf.summary.histogram(var.op.name, var)
            sz = var.get_shape().as_list()
            if len(sz) == 4 and sz[2] == 3:
                kernels = tf.transpose(var, [3, 0, 1, 2])
                tf.summary.image(var.op.name + '/kernels',
                                 group_batch_images(kernels), max_outputs=1)
    for activation in activation_list:
        if activation is not None:
            tf.summary.histogram(activation.op.name +
                                 '/activations', activation)
            #tf.summary.scalar(activation.op.name + '/sparsity', tf.nn.zero_fraction(activation))


def _learning_rate_decay_fn(learning_rate, global_step):
  return tf.train.exponential_decay(
      learning_rate,
      global_step,
      decay_steps=FLAGS.decay_steps ,
      decay_rate=FLAGS.decay_rate,
      staircase=True)

learning_rate_decay_fn = _learning_rate_decay_fn

def train(model, data,
          batch_size=128,
          learning_rate=FLAGS.learning_rate,
          log_dir='./log',
          checkpoint_dir='./checkpoint',
          num_epochs=-1):

    # tf Graph input
    with tf.device('/cpu:0'):
        with tf.name_scope('data'):
            if FLAGS.dataset == "imagenet" :
                x, yt =image_processing.distorted_inputs(data,batch_size=batch_size,num_preprocess_threads=FLAGS.num_threads)
            else :
                x, yt = data.generate_batches(batch_size,num_threads=FLAGS.num_threads)
        global_step =  tf.get_variable('global_step', shape=[], dtype=tf.int64,
                             initializer=tf.constant_initializer(0),
                             trainable=False)
    if FLAGS.gpu:
        device_str='/gpu:' + str(FLAGS.device)
    else:
        device_str='/cpu:0'
    with tf.device(device_str):
        y = model(x, is_training=True)
        # Define loss and optimizer
        with tf.name_scope('objective'):
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=yt, logits=y))
            accuracy = tf.reduce_mean(
                tf.cast(tf.nn.in_top_k(y, yt, 1), tf.float32))
        opt = tf.contrib.layers.optimize_loss(loss, global_step, learning_rate, 'Adam',
                                              gradient_noise_scale=None, gradient_multipliers=None,
                                              clip_gradients=None, #moving_average_decay=0.9,
                                              learning_rate_decay_fn=learning_rate_decay_fn if FLAGS.using_learning_rate_decay_fn else None,
                                              update_ops=None, variables=None, name=None)
        #grads = opt.compute_gradients(loss)
        #apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # loss_avg

    ema = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step, name='average')
    ema_op = ema.apply([loss, accuracy] + tf.trainable_variables())
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_op)

    loss_avg = ema.average(loss)
    tf.summary.scalar('loss/training', loss_avg)
    accuracy_avg = ema.average(accuracy)
    tf.summary.scalar('accuracy/training', accuracy_avg)

    check_loss = tf.check_numerics(loss, 'model diverged: loss->nan')
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, check_loss)
    updates_collection = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies([opt]):
        train_op = tf.group(*updates_collection)

    if FLAGS.summary:
        add_summaries( scalar_list=[accuracy, accuracy_avg, loss, loss_avg],
            activation_list=tf.get_collection(tf.GraphKeys.ACTIVATIONS),
            var_list=tf.trainable_variables())
            # grad_list=grads)

    summary_op = tf.summary.merge_all()
    # Configure options for session
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.InteractiveSession(
        config=tf.ConfigProto(
            log_device_placement=False,
            allow_soft_placement=True,
            gpu_options=gpu_options,
        )
    )
    if FLAGS.resume:
      logging.info('resuming from '+checkpoint_dir)
      saver = tf.train.Saver()
      ckpt = tf.train.get_checkpoint_state(checkpoint_dir+'/')
      if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
      else:
        print('No checkpoint file found')
        return
      #print sess.run('global_step:0')
      #print global_step.eval()
    else:
      saver = tf.train.Saver(max_to_keep=5)
      sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    num_batches = data.size[0] / batch_size
    summary_writer = tf.summary.FileWriter(log_dir, graph=sess.graph)
    epoch = global_step.eval()/num_batches if FLAGS.resume else 0
    display_interval=FLAGS.display_interval or num_batches/10
    test_interval=FLAGS.test_interval or num_batches/2
    logging.info('num of trainable paramaters: %d' %count_params(tf.trainable_variables()))
    while epoch != num_epochs:
        epoch += 1
        curr_step = 0
        # Initializing the variables

        #with tf.Session() as session:
        #    print(session.run(ww))

        logging.info('Started epoch %d' % epoch)
        bar = Bar('Training', max=num_batches,
                  suffix='%(percent)d%% eta: %(eta)ds')
        while curr_step < data.size[0]:
            _, loss_val,step= sess.run(
              [train_op,loss,global_step])
            if step%display_interval==0:
              step, acc_value, loss_value, summary = sess.run(
                [global_step, accuracy_avg, loss_avg, summary_op])
              logging.info("step %d loss %.3f accuracy %.3f" %(step,loss_value,acc_value))
              summary_out = tf.Summary()
              summary_out.ParseFromString(summary)
              summary_writer.add_summary(summary_out, step)
              summary_writer.flush()
            if step%test_interval==0:
              saver.save(sess, save_path=checkpoint_dir +
                   '/model.ckpt', global_step=global_step)
              test_top1,test_top5,test_loss = evaluate(model, FLAGS.dataset,
                                       batch_size=batch_size,
                                       checkpoint_dir=checkpoint_dir)
              logging.info('Test loss %.3f Test top1 %.3f Test top5 %.3f' % (test_loss,test_top1,test_top5))
              summary_out = tf.Summary()
              summary_out.ParseFromString(summary)
              summary_out.value.add(tag='accuracy/test_top1', simple_value=test_top1)
              summary_out.value.add(tag='accuracy/test_top5', simple_value=test_top5)
              summary_out.value.add(tag='loss/test', simple_value=test_loss)
              summary_writer.add_summary(summary_out, step)
              summary_writer.flush()
            curr_step += FLAGS.batch_size
            bar.next()

        bar.finish()
        logging.info("Finished epoch %d " %epoch)

    # When done, ask the threads to stop.
    coord.request_stop()
    coord.join(threads)
    coord.clear_stop()
    summary_writer.close()


def main(argv=None):  # pylint: disable=unused-argument
    if not gfile.Exists(FLAGS.checkpoint_dir):
        # gfile.DeleteRecursively(FLAGS.checkpoint_dir)
        gfile.MakeDirs(FLAGS.checkpoint_dir)
        model_file = os.path.join('models', FLAGS.model + '.py')
        assert gfile.Exists(model_file), 'no model file named: ' + model_file
        gfile.Copy(model_file, FLAGS.checkpoint_dir + '/model.py')
    logInit(FLAGS.loggingFile,resume=FLAGS.resume)
    m = importlib.import_module('.' + FLAGS.model, 'models')
    data = get_data_provider(FLAGS.dataset, training=True)

    train(m.model, data,
          batch_size=FLAGS.batch_size,
          checkpoint_dir=FLAGS.checkpoint_dir,
          log_dir=FLAGS.log_dir,
          num_epochs=FLAGS.num_epochs)


if __name__ == '__main__':
    tf.app.run()
