import os

a = '../data/audio'
b = '../data/10001'

files = os.listdir(a)

for file in files:
    name = file.split('_')[0]
    full_path = os.path.join(b, name + '_P')
    cmd = 'mkdir ' + full_path
    os.system(cmd)
    cmd = 'cp ' + os.path.join(a, file) + ' ' + full_path
    os.system(cmd)
