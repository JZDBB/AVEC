import os
FPS = 30
data_dir = '../data/10001/'
dirs = os.listdir(data_dir)
max_frame = 0
for d in dirs:
    curr = os.path.join(data_dir, d)
    name = d.split('_')[0]
    csv_file = name + '_TRANSCRIPT.csv'
    out_file = name + '_WORD.csv'
    of = open(os.path.join(curr, out_file), 'w')
    with open(os.path.join(curr, csv_file), 'r') as f:
        f.readline()
        while True:
            l = f.readline().lower().replace('\n', '').replace('\r', '').split('\t')
            if l == ['']:
                break
            try:
                start_time = float(l[0])
                stop_time = float(l[1])
            except:
                a = 1
            speaker = l[2]
            value = l[3]
            if speaker == 'ellie':
                continue
            w = value.strip().split(' ')
            n = len(w)
            start_frame = int(start_time * FPS)
            stop_frame = int(stop_time * FPS)
            if start_frame <= max_frame:
                start_frame = max_frame + 1
            max_frame = stop_frame
            if stop_frame <= start_frame:
                print(csv_file, start_time)

            d = (stop_frame - start_frame) / n
            if d == 0:
                a = 1
            for i in range(start_frame, stop_frame + 1):
                wid = int((i - start_frame) / d)
                if wid == n: wid = n -1
                of.write("%d, %s\n"%(i, w[wid]))
    of.close()
