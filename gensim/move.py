import os
import shutil

a = '../data/transcripts/transcripts'
b = '../data/10001'

files = os.listdir(a)

for file in files:
    name = file.split('_')[0]
    full_path = os.path.join(b, name + '_P')
    # Windows下操作
    os.mkdir(full_path)
    shutil.copy(os.path.join(a, file), full_path)

    # ubuntu下操作
    # cmd = 'mkdir ' + full_path
    # os.system(cmd)
    # cmd = 'cp ' + os.path.join(a, file) + ' ' + full_path
    # os.system(cmd)
