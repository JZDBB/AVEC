import os
labels = {}
p = 0
n = 0
with open('/home/yqi/Downloads/BaiduDownloads/DAIC/labels/dev_split_Depression_AVEC2017.csv') as f:
    l = f.readline()
    while True:
        l = f.readline()
        if l == '':
            break
        id, label = l.split(',')[0:2]
        label = int(label)
        labels[id] = label
        if label == 0:
            n += 1
        else:
            p += 1

print(p, n)

n = 0
for k in labels.keys():
    path = '/data/10001/%s_P/%s_LOGMEL.csv' % (k, k)
    try:
        n += len(open(path, 'rU').readlines()) // 249
    except:
        continue

print(n)