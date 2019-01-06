from tcn import TemporalConvNet as TCN
import tensorflow as tf
import os
from oi.panoreader import PANOReader
import time
from wnconv1d import WEIGHT_DECAY
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


if __name__ == '__main__':
    net = TCN([128, 128, 128, 128, 128], stride=1, kernel_size=5, dropout=0.5)
    data_dir = '/home/yqi/data/10000/valid'
    filenames = tf.train.match_filenames_once(os.path.join(data_dir, '*.records'))
    mel, lin, label, PHQ = PANOReader(filenames, 16, 32, 32, num_epochs=100, drop_remainder=False).read()
    mel = tf.contrib.layers.batch_norm(mel, is_training=net.is_training)
    lin = tf.contrib.layers.batch_norm(lin, is_training=net.is_training)
    batch_norm_update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    x = tf.concat([mel, lin], axis=-1)
    feat = tf.reshape(net(x), (-1, 128))
    logits = tf.contrib.layers.fully_connected(feat, 2, None,
                                               weights_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY))
    sc = tf.contrib.layers.fully_connected(feat, 26, None,
                                           weights_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY))
    tf.summary.histogram("sc", sc)
    weights = tf.where(tf.equal(label, 0), tf.ones_like(label), tf.ones_like(label))
    loss = tf.losses.sparse_softmax_cross_entropy(label, logits, weights=weights)
    tf.summary.scalar("loss/cls_loss", loss)
    regress = tf.losses.sparse_softmax_cross_entropy(tf.to_int64(PHQ), sc)
    tf.summary.scalar("loss/regress_loss", regress)
    reg = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    tf.summary.scalar("loss/reg_loss", reg)
    loss += reg
    loss += regress
    tf.summary.scalar("loss/total_loss", loss)
    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(0.001)

    predictions = tf.argmax(logits, axis=-1, name="predictions")
    precision, prediction_update_op = tf.metrics.precision(labels=label, predictions=predictions,
                                                           name="metric_precision")
    recall, recall_update_op = tf.metrics.recall(labels=label, predictions=predictions,
                                                 name="metric_recall")
    accuracy, accuracy_update_op = tf.metrics.accuracy(labels=label, predictions=predictions,
                                                       name="metric_accuracy")
    with tf.control_dependencies([recall_update_op, prediction_update_op]):
        f1_score = 2 * precision * recall / (precision + recall)
    with tf.name_scope('scores'):
        tf.summary.scalar("accuracy", accuracy)
        tf.summary.scalar("recall", recall)
        tf.summary.scalar("precision", precision)
        tf.summary.scalar("F1_score", f1_score)
    summ = tf.summary.merge_all()

    saver = tf.train.Saver(max_to_keep=20)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.get_collection(tf.GraphKeys.INIT_OP))
        valid_writer = tf.summary.FileWriter("logs/valid", sess.graph)

        curr = None
        while True:
            latest = tf.train.latest_checkpoint('ckpt')
            if latest != curr:
                g = int(latest.split('-')[1]) if latest is not None else 0
                saver.restore(sess, latest)
                s, _ = sess.run([summ, accuracy_update_op], feed_dict={net.is_training: False})
                valid_writer.add_summary(s, g)
                valid_writer.flush()
            curr = latest
            time.sleep(1)