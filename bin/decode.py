#!/usr/bin/python

"""
Script to engage the decoding pipeline. Expects:

    -- pacfile: A PAC-format lossy-compressed audio binary
    -- outfile: The decoded 16-bit 44100Hz sampled PCM WAV file
"""

import sys
import time
import argparse

import numpy as np

from codec.pipeline import Decoder
from codec.pacfile import PACReader
from codec.wav_util import write_wav_data


# Use argparse to get the filename. Additionally allow the program to print
# out a help block at the command line when invoked with '-h'
parser = argparse.ArgumentParser(description='An experimental audio coder.')
parser.add_argument('infile', metavar='pacfile', type=str,
                    help='The filename of the PACFile to decode.')
parser.add_argument('outfile', metavar='outfile', type=str,
                    help='The filename to write the 16bit WAV file to.')  
args = parser.parse_args()


def run_decoder():
    """ Move data around between the PAC reader and the decoding pipeline. """
    print 'Starting the decoder! You\'re gonna hear some serious shit!'

    with PACReader(args.infile) as pac:
        decoder = Decoder(pac.metadata())
        data = [np.array([], dtype=np.float64)] * decoder.channels
        done = False
        while not done:

            # 1. Acquire the next block of encoded data
            block, window = pac.read_block()
            
            # 2. Print the cute window state indicator. Helps debug f'sho
            print (window_state_char(window)), 

            # 3. Give the block of data to the decoder
            wav_data = decoder.decode_block(block, window)

            # 4. Persist this data to write out to the WAV file later
            if wav_data:
                for ch in range(decoder.channels):
                    data[ch] = np.concatenate([data[ch], wav_data[ch]])
            else:
                done = True

            # 5. Necessary to print while we are doing decoding
            sys.stdout.flush()

    print
    print 'Finished. Writing the WAV file...'
    write_wav_data(args.outfile, data)


def window_state_char(window):
    """ Helper routine to print to stdout. """
    if window == 0:
        return '_'
    elif window == 1:
        return '/'
    elif window == 2:
        return "^"
    elif window == 3:
        return "\\"
    else:
        return "!"


if __name__ == '__main__':
    run_decoder()
