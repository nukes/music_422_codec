''' Utility functions to write out WAV files '''

import struct

import numpy as np
from codec.quantize import vQuantizeUniform


def write_wav_data(filename, data):
    ''' Take a filename and stereo data array (2, samples) and write
    out the WAV file for it.
    '''

    # Figure out how many samples and the total number of bytes writing out
    samples = min([len(ch) for ch in data])
    bytes = 2 * samples * (16/8)

    # Convert the data we received into 2's complement binary
    # TODO: Refactor into a for...each
    pcm_data = []
    for ch in range(2):
        d = data[ch]
        signs = np.signbit(d)
        d[signs] *= -1
        d = vQuantizeUniform(d, 16).astype(np.int16)
        d[signs] *= -1
        pcm_data.append(d)   

    # WAV expects the stereo data to be a signle array of interleaved
    # data for each channel.
    block = [pcm_data[ch][sam] for sam in xrange(samples) for ch in range(2)]
    block = np.asarray(block, dtype=np.int16)

    # Finally, create this disgusting header data
    header = struct.pack('<4sL4s4sLHHLLHH4sL', 'RIFF',  36 + bytes, 'WAVE', 'fmt ', 16,
                         1, 2, 44100, 44100*2*(16/8), 2*(16/8), 16, 'data', bytes)

    f = open('test_out.wav', 'wb')
    f.write(header)
    f.write(block.tostring())
    f.close()
