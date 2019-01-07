import tensorflow as tf

from tcn import TemporalConvNet as TCN

import os
from oi.panoreader import PANOReader
import time
from wnconv1d import WEIGHT_DECAY
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"


if __name__ == '__main__':

    data_dir = '../data/10000/valid'
    for name in os.listdir(data_dir):
        tf.reset_default_graph()
        tf.logging.set_verbosity(tf.logging.ERROR)
        net = TCN([128, 128, 128, 128, 128], stride=1, kernel_size=5, dropout=0.7)
        filenames = tf.train.match_filenames_once(os.path.join(data_dir, name))
        mel, lin, label, PHQ = PANOReader(filenames, 128, 1, 1, num_epochs=1, drop_remainder=False).read()
        mel = tf.contrib.layers.batch_norm(mel, is_training=net.is_training)
        lin = tf.contrib.layers.batch_norm(lin, is_training=net.is_training)
        batch_norm_update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        x = tf.concat([mel, lin], axis=-1)
        feat = tf.reshape(net(x), (-1, 128))
        logits = tf.contrib.layers.fully_connected(feat, 2, None,
                                                   weights_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY))

        predictions = tf.argmax(logits, axis=-1, name="predictions")

        accuracy, accuracy_update_op = tf.metrics.accuracy(labels=label, predictions=predictions,
                                                           name="metric_accuracy")

        saver = tf.train.Saver(max_to_keep=20)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(tf.get_collection(tf.GraphKeys.INIT_OP))
            latest = tf.train.latest_checkpoint('ckpt')
            saver.restore(sess, latest)

            while True:
                try:
                    l, p, _ = sess.run([label, predictions, accuracy_update_op], feed_dict={net.is_training: False})
                except:
                    break
            a = sess.run(accuracy, feed_dict={net.is_training: False})
            print("%s, %d, %f"%(name, l[0], a))
