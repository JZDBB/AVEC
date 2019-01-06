import os
from sklearn.externals import joblib
model = joblib.load('word.pkl')

data_dir = '../data/10001/'
dirs = os.listdir(data_dir)
for d in dirs:
    curr = os.path.join(data_dir, d)
    name = d.split('_')[0]
    mel_file = name + '_LOGMEL.csv'
    word_file = name + '_WORD.csv'
    word = {}
    with open(os.path.join(curr, word_file)) as f:
        while True:
            l = f.readline().strip('\n').split(',')
            if l == ['']:
                break
            word[int(l[0])] = l[1].strip(' ')

    keys = word.keys()
    feature3_file = name + '_LINGUISTIC.csv'
    n = len(open(os.path.join(curr, mel_file), 'rU').readlines())
    with open(os.path.join(curr, feature3_file), 'w') as f:
        for i in range(1, n + 1):
            if i in keys:
                for j in range(300):
                    f.write('%f,'%model[word[i]][j])
                    # f.write(',')
                f.write('\n')
            else:
                for j in range(300):
                    f.write('0,')
                    # f.write(',')
                f.write('\n')
