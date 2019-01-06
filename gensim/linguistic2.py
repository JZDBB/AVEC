import os

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
max_frame = 0
of = open('WORD.csv', 'w')
for d in dirs:
    curr = os.path.join(data_dir, d)
    name = d.split('_')[0]
    if not name in labels.keys():
        continue
    csv_file = name + '_TRANSCRIPT.csv'

    with open(os.path.join(curr, csv_file), 'r') as f:
        f.readline()
        while True:
            l = f.readline().lower().replace('\n', '').replace('\r', '').split('\t')
            if l == ['']:
                break
            speaker = l[2]
            value = l[3]
            if speaker == 'ellie':
                continue
            if value == '':
                continue
            # content.append('%d,%d,%f,%s\n' % (labels[name], PHQs[name],
            #                                   (float(l[1]) - float(l[0]))/len(value.split(' ')), value))
            of.write('%s,%d,%d,%f,%s\n' % (name, labels[name], PHQs[name], (float(l[1]) - float(l[0]))/len(value.split(' ')), value))

of.close()