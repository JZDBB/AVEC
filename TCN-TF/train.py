from tcn import TemporalConvNet as TCN
import tensorflow as tf
import os
from oi.panoreader import PANOReader
from wnconv1d import WEIGHT_DECAY

if __name__ == '__main__':
    net = TCN([128, 128, 128, 128, 128], stride=1, kernel_size=5, dropout=0.5)
    data_dir = '../data/10000/train'
    filenames = tf.train.match_filenames_once(os.path.join(data_dir, '*.records'))
    mel, lin, label, PHQ = PANOReader(filenames, 16, 32, 32, num_epochs=None, drop_remainder=False).read()
    mel = tf.contrib.layers.batch_norm(mel, is_training=net.is_training)
    lin = tf.contrib.layers.batch_norm(lin, is_training=net.is_training)
    batch_norm_update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # x = tf.concat([mel, lin], axis=-1)
    x = lin
    feat = tf.reshape(net(x), (-1, 128))
    logits = tf.contrib.layers.fully_connected(feat, 2, None, weights_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY))
    sc = tf.contrib.layers.fully_connected(feat, 26, None, weights_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY))
    tf.summary.histogram("sc", sc)
    weights = tf.where(tf.equal(label, 0), tf.ones_like(label), tf.ones_like(label))#
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
    with tf.control_dependencies(batch_norm_update_op):
        train_op = optimizer.minimize(loss, global_step=global_step)
    saver = tf.train.Saver(max_to_keep=20)

    latest = tf.train.latest_checkpoint('ckpt')
    g = int(latest.split('-')[1]) if latest is not None else 0

    with tf.Session() as sess:
        ckpt = os.path.join('ckpt', "model.ckpt")
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.get_collection(tf.GraphKeys.INIT_OP))
        train_writer = tf.summary.FileWriter("logs/train", sess.graph)
        if g > 0:
            saver.restore(sess, latest)
            print('load ckpt %d' % g)
        n = g
        while True:
            sess.run(train_op, feed_dict={net.is_training: True})
            n += 1
            if n % 100 == 0:
                saver.save(sess, ckpt, global_step=n)
                s, _ = sess.run([summ, accuracy_update_op], feed_dict={net.is_training: False})
                train_writer.add_summary(s, n)
                train_writer.flush()
