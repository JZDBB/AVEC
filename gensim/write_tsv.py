from sklearn.externals import joblib
a = joblib.load('word.pkl')
with open('test.tsv', 'w') as f:
    for k in a.keys():
        for i in range(300):
            f.write('%f\t'% a[k][i])
        f.write('\n')

with open('metadata.tsv', 'w') as f:
    f.write('word\tindex\n')
    for i, k in enumerate(a.keys()):
        # f.write('%d\t%s\n'%(i, k))
        f.write('%s\t%d\n' % (k, i+1))
