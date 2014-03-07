'''
Class to wrap the PAC file as specified by Drs. Bosi and Goldberg.
'''

import struct

import numpy as np

from provided.audiofile import CodingParams
from provided.bitpack import PackedBits
from provided.pcodec import Encode, EncodeSingleChannel
from codec.psychoac import ScaleFactorBands, AssignMDCTLinesFromFreqLimits


class PACFile(object):

    def __init__(self, filename):
        self.filename = filename
        self.tag = 'PAC '

    def write_header(self, sample_rate, channels, samples, mdct_lines, scale_bits, mant_bits):
        ''' Accept a lot of bullshit to write the header out. '''
        self.file.write(self.tag)

        # Ensure that the number of samples in the file is a multiple of the
        # number of MDCT half-block size (i.e. N/2, the positive spectrum) 
        # and zero pad it as needed.
        if not samples % mdct_lines:
            samples += (mdct_lines - samples % mdct_lines)

        # Add in another delay block for the overlap-and-add reconstruction
        samples += mdct_lines

        # Write out the file attributes
        header = struct.pack('<LHLLHH', sample_rate, channels, samples, \
                                        mdct_lines, scale_bits, mant_bits)
        self.file.write(header)

        # Write out to the file the scale factor allocations
        alloc = AssignMDCTLinesFromFreqLimits(mdct_lines, sample_rate)
        self.scale_factors = ScaleFactorBands(alloc)
        band_coding = struct.pack('<L', self.scale_factors.nBands)
        line_coding = struct.pack('<' + str(self.scale_factors.nBands) + 'H',
                                  *(self.scale_factors.nLines.tolist()))
        self.file.write(band_coding)
        self.file.write(line_coding)

        # Start with the first block of data for the MDCT block
        self.previous_block = []
        for ch in range(channels):
            self.previous_block.append(np.zeros(mdct_lines, dtype=np.float64))

    def write_data(self, data, sample_rate, channels, samples, mdct_lines, scale_bits, mant_bits, target_bps):
        ''' Write a block of signed float data in the range of -1...1 to the
        PACfile the object holds a reference to.
        '''

        # Construct the full block to write out
        full_block = []
        for ch in range(channels):
            block = np.concatenate([self.previous_block[ch], data[ch]])
            full_block.append(block)
        self.previous_block = data

        # TODO: Remove the CodingParams once they are no longer needed
        cp = CodingParams()
        cp.sampleRate = sample_rate
        cp.nChannels = channels
        cp.nMDCTLines = mdct_lines
        cp.nScaleBits = scale_bits
        cp.nMantSizeBits = mant_bits
        cp.sfBands = self.scale_factors
        cp.targetBitsPerSample = target_bps

        # TODO: Move this out of the file. This is retarded.
        (scale_factor, bit_alloc, mant, overall_scale) = Encode(full_block, cp)

        # Write the encoded data to the file
        for ch in range(channels):

            # Determine the size of the channel's block
            bits = scale_bits
            for band in range(self.scale_factors.nBands):
                bits += mant_bits + scale_bits
                if bit_alloc[ch][band]:
                    bits += bit_alloc[ch][band] * self.scale_factors.nLines[band]

            # TODO: This is where we count the bits needed for block switching
            # i.e. This is where we add '2' to the count of bits needed

            # Convert the bits to bytes, using the conventional definition of
            # 8 bits to 1 byte.  Add a "spillover" byte if we are just shy of
            # the 8 byte boundary -- i.e. guarantee we have the byte capacity
            bytes = bits / 8 + int(bits % 8 != 0)

            # Write out the size of the data block
            out = struct.pack('<L', int(bytes))
            self.file.write(out)

            # TODO: Eliminate this PackedBits if possible. Seems hard, though.
            pb = PackedBits()
            pb.Size(bytes)

            # Actually pack the data
            pb.WriteBits(overall_scale[ch], scale_bits)
            i_mant = 0
            for band in range(self.scale_factors.nBands):
                alloc = bit_alloc[ch][band]
                if alloc:
                    alloc -= 1
                pb.WriteBits(alloc, mant_bits)
                pb.WriteBits(scale_factor[ch][band], scale_bits)
                if bit_alloc[ch][band]:
                    for i in range(self.scale_factors.nLines[band]):
                        pb.WriteBits(mant[ch][i_mant+i], bit_alloc[ch][band])
                    i_mant += self.scale_factors.nLines[band]

            # TODO: This is where we would pack in the block switching data

            # Finally, write this damned data to the file
            self.file.write(pb.GetPackedData())


    def open_to_write(self):
        self.file = open(self.filename, 'wb')

    def close(self):
        self.file.close()

