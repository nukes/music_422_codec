#!/usr/bin/python

"""
Script to engage the encoding pipeline. Expects:

    -- wavfile: A 16-bit, 44100Hz sampled PCM WAV file
    -- outfile: A PAC-format lossy-compressed audio binary
"""

import sys
import argparse

from codec.pacfile import PACWriter
from codec.transbuffer import TransientBuffer
from codec.pipeline import Encoder


# Use argparse to get the filename. 
# Additionally allow the program to print out a help block at the command
# line when invoked with '-h'
parser = argparse.ArgumentParser(description='An experimental audio coder.')
parser.add_argument('infile', metavar='wavfile', type=str,
                    help='The filename of the 16bit PCM WAV to encode.')
parser.add_argument('outfile', metavar='outfile', type=str,
                    help='The filename to write out the encoding file.')  
args = parser.parse_args()


def run_encoder():
    """ Move data around between the file reader and the encoding pipeline. """
    # Set up the basic, hardcoded parameters for the WAV file.
    # It would make sense to pull these out into command line options at some point
    channels = 2
    sample_rate = 44100
    mdct_lines = 512
    scale_bits = 3
    mant_bits = 4
    target_bps = 2.9

    # Open up a file reference to the file we want to write
    buf = TransientBuffer(args.infile)
    wav_data, window = buf.next()
    encoder = Encoder(sample_rate, channels, mdct_lines, scale_bits, mant_bits, target_bps)

    print 'Starting the encoder! Wham bam thank you ma\'am!'
    with PACWriter(args.outfile, sample_rate, channels, buf._samples, mdct_lines, scale_bits, mant_bits) as pac:
        while len(wav_data) > 0:

            # 1. Print cute little text based on the the window state. Helps debug!
            print (window_state_char(window)),

            # 2. Send the data block and window state to the encoder pipeline
            (block_size, block_data) = encoder.encode_block(wav_data, window)

            # 3. Pass these bits to the PACFile to write out
            pac.write_data(block_size, block_data)

            # 4. Acquire the next data block and its corresponding window state
            wav_data, window = buf.next()

            # 5. Necessary to print while we are doing the encoding
            sys.stdout.flush()


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
    run_encoder()
