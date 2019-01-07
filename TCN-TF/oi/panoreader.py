import tensorflow as tf
import os
import cv2
import numpy as np
from TCN.oi import tfrecordsreader


class PANOReader(tfrecordsreader.TFRecordsReader):

    def __init__(self,
                 filenames,
                 batch_size,
                 num_readers,
                 read_threads,
                 num_epochs=None,
                 drop_remainder=True,
                 shuffle=True,
                 fake=False, **kwargs):
        super(PANOReader, self).__init__(filenames,
                                         batch_size,
                                         num_readers,
                                         read_threads,
                                         num_epochs,
                                         drop_remainder,
                                         shuffle,
                                         fake, **kwargs)

    def _parser(self, record):
        tfrecord_features = tf.parse_single_example(record,
                                                    features={
                                                        'mel': tf.FixedLenFeature([249, 240], dtype=tf.float32),
                                                        'lin': tf.FixedLenFeature([249, 300], dtype=tf.float32),
                                                        'label': tf.FixedLenFeature([], dtype=tf.int64),
                                                        'PHQ': tf.FixedLenFeature([], dtype=tf.float32)},
                                                    name='features')
        mel = tfrecord_features['mel']
        lin = tfrecord_features['lin']
        label = tfrecord_features['label']
        PHQ = tfrecord_features['PHQ']
        return mel, lin, label, PHQ

    def _post_process(self, iterator):
        mel, lin, label, PHQ = iterator.get_next()
        return mel, lin, label, PHQ

if __name__ == '__main__':
    data_dir = '/home/yqi/data/10001/300_P'
    filenames = tf.train.match_filenames_once(os.path.join(data_dir, '*.records'))
    mel, lin, label = PANOReader(filenames, 1, 10, 4, num_epochs=1, drop_remainder=False).read()

    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()
        a = tf.get_collection(tf.GraphKeys.INIT_OP)
        sess.run(tf.get_collection(tf.GraphKeys.INIT_OP))
        n = 0
        while 1:
            n += 1
            try:
                i, l = sess.run([images, labels])
                print(i)
                # print os.path.join('../' + str(l[0]), str(n) + '.jpg')
                cv2.imwrite(os.path.join('../images2/' + str(l[0,]), str(n) + '.jpg'), i[0, :].astype(np.uint8))
            except Exception as e:
                print(e)
                break
