#!/usr/bin/python

import sys
import time

from codec.transbuffer import TransientBuffer
from codec.pacfile import PACWriter

elapsed = time.time()

buf = TransientBuffer('SQAM/harp40_1.wav')

channels = 2
sample_rate = 44100
samples = buf._samples
mdct_lines = 512
scale_bits = 3
mant_bits = 4
target_bps = 2.9

pac = PACWriter('coded.pac', sample_rate, channels, samples, mdct_lines, scale_bits, mant_bits, target_bps)

print 'Beginning the Encoder!'
i = 0
done = False
while not done:
    x, win = buf.next()
    if len(x) == 0:
        done = True
    else:
        i += 1
        if win == 0:
            print('_'),
        elif win == 1:
            print('/'),
        elif win == 2:
            print("^"),
        elif win == 3:
            print("\\"),
        pac.write_data(x, win)
        sys.stdout.flush()

pac.close()

print
print "======"
print "Detections: ", buf._detects

elapsed = time.time() - elapsed
print "Done, in", elapsed, "seconds."
print
