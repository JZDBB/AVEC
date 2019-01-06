import os
labels = {}
with open('/home/yqi/Downloads/BaiduDownloads/DAIC/labels/train_split_Depression_AVEC2017.csv') as f:
    l = f.readline()
    while True:
        l = f.readline()
        if l == '':
            break
        id, label = l.split(',')[0:2]
        label = int(label)
        labels[id] = label

for k in labels.keys():
    cmd = 'ln -s /data/10001/%s_P/%s_TF.records /data/10000/train/%s_TF.records' % (k, k, k)
    os.system(cmd)

labels = {}
with open('/home/yqi/Downloads/BaiduDownloads/DAIC/labels/dev_split_Depression_AVEC2017.csv') as f:
    l = f.readline()
    while True:
        l = f.readline()
        if l == '':
            break
        id, label = l.split(',')[0:2]
        label = int(label)
        labels[id] = label

for k in labels.keys():
    cmd = 'ln -s /data/10001/%s_P/%s_TF.records /data/10000/valid/%s_TF.records' % (k, k, k)
    os.system(cmd)