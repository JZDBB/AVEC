from __future__ import print_function
from __future__ import division
import tensorflow as tf
import os


class converter(object):

    def __init__(self):
        pass


    def tolist(self, value):
        if not isinstance(value, list):
            value = [value]
        return value

    def __bytes_feature(self, value):
        value = self.tolist(value)
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def __int64_feature(self, value):
        value = self.tolist(value)
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def __float_feature(self, value):
        value = self.tolist(value)
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def convert(self):
        labels = {}
        PHQs = {}
        with open('../data/labels/train_split_Depression_AVEC2017.csv') as f:
            l = f.readline()
            while True:
                l = f.readline()
                if l == '':
                    break
                id, label, PHQ = l.split(',')[0:3]
                label = int(label)
                PHQ = float(PHQ)
                labels[id] = label
                PHQs[id] = PHQ

        with open('../data/labels/dev_split_Depression_AVEC2017.csv') as f:
            l = f.readline()
            while True:
                l = f.readline()
                if l == '':
                    break
                id, label, PHQ = l.split(',')[0:3]
                label = int(label)
                PHQ = float(PHQ)
                labels[id] = label
                PHQs[id] = PHQ
        data_dir = '../data/10001/'
        dirs = os.listdir(data_dir)
        for dd in labels.keys():
            print(dd)
            d = dd + '_P'
            if not d in dirs:
                continue
            curr = os.path.join(data_dir, d)
            name = d.split('_')[0]
            lin_file = name + '_LINGUISTIC.csv'
            mel_file = name + '_LOGMEL.csv'
            records_file = name + '_TF.records'
            f_lin = open(os.path.join(curr, lin_file))
            f_mel = open(os.path.join(curr, mel_file))
            writer = tf.python_io.TFRecordWriter(os.path.join(curr, records_file))
            lin = f_lin.read().split('\n')[:-1]
            mel = f_mel.read().split('\n')[:-1]
            frames = []
            ls = []
            ms = []
            for i in range(len(lin) // 249 * 249):
                ls.extend([float(l) for l in lin[i].split(',')[:-1]])
                a = [float(m) for m in mel[i].split(',')[:-1]]
                if ls[0] == 0:
                    ms.extend([0.]*240)
                else:
                    ms.extend([float(m) for m in mel[i].split(',')[:-1]])
                if i % 249 == 248:
                    num_zero = 0
                    for l in ls:
                        if l == 0:
                            num_zero += 1
                    if num_zero / len(ls) < 0.8:
                        ls = []
                        ms = []
                        continue
                    frames.append({'mel': ms, 'lin': ls})
                    ls = []
                    ms = []
            for f in frames:
                example = tf.train.Example(features=tf.train.Features(feature={
                    'mel': self.__float_feature(f['mel']),
                    'lin': self.__float_feature(f['lin']),
                    'label': self.__int64_feature(labels[name]),
                    'PHQ': self.__float_feature(PHQs[name])
                }))
                writer.write(example.SerializeToString())

            f_lin.close()
            f_mel.close()
            writer.close()


def main():
    c = converter()
    c.convert()

if __name__ == '__main__':
    main()

