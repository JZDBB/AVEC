import librosa
# import matplotlib.pyplot as plt
import numpy as np
# import librosa.display
import os

data_dir = 'data/10001/'
dirs = os.listdir(data_dir)
for d in dirs:
    print(d)
    curr = os.path.join(data_dir, d)
    name = d.split('_')[0]
    wav_file = name + '_AUDIO.wav'
    mel_file = name + '_LOGMEL.csv'
    y, sr = librosa.load(os.path.join(curr, wav_file), sr=16000)
    y = librosa.amplitude_to_db(y)
    y = np.minimum(y, -15)
    y = librosa.db_to_amplitude(y)
    y = y + np.min(y)
    y = y / np.max(y)

    f = open(os.path.join(curr, mel_file), 'w')
    n_y = len(y)
    n = 1
    while 533.33333 * n < n_y:
        yy = y[int(np.around(533.333333*(n - 1))):int(np.around(533.333333*n))]
        s = librosa.power_to_db(librosa.feature.melspectrogram(y=yy, sr=sr, n_mels=80, fmax=8000, hop_length=int(len(yy)/3 + 1)))
        s = np.reshape(s, (-1))
        for i in range(s.shape[0]):
            f.write('%.05f,' % s[i])
        f.write('\n')
        n += 1
    f.close()
