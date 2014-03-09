#!/usr/bin/python

import sys
import time

import numpy as np

from codec.pacfile import PACReader
from codec.wav_util import write_wav_data


elapsed = time.time()

pac = PACReader('coded.pac')

print 'Starting the decoder! You\'re gonna hear some serious shit!'
i = 0
done = False
detects = 0
data = [np.array([], dtype=np.float64), np.array([], dtype=np.float64)]
while not done:
    x, win_state = pac.read_block()
    if win_state == 0:
        print('_'),
    elif win_state == 1:
        print('/'),
    elif win_state == 2:
        print("^"),
    elif win_state == 3:
        print("\\"),

    if win_state == 1:
        detects += 1
    if not x:
        done = True
    else:
        i += 1
        data[0] = np.concatenate([data[0], x[0]])
        data[1] = np.concatenate([data[1], x[1]])
        sys.stdout.flush()

pac.close()

print
print "======"
print "Detections: ", detects

print 'Writing the WAV file...'
write_wav_data('output.wav', data)

elapsed = time.time() - elapsed
print "Done, in", elapsed, "seconds."
