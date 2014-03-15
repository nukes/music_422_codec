"""
Class to wrap the PAC file as specified by Drs. Bosi and Goldberg.
"""

import struct

import numpy as np

from codec.bitpack import PackedBits
from codec.psychoac import AssignMDCTLinesFromFreqLimits, ScaleFactorBands


class PACWriter(object):

    def __init__(self, filename, sample_rate, channels, samples, mdct_lines, scale_bits, mant_bits):
        """ Provide parameters to fully specify a PAC file. The file is opened
        and the header is written upon initialization. """
        self.file = open(filename, 'wb')
        self.file.write('PAC ')

        # Ensure that the number of samples in the file is a multiple of the
        # number of MDCT half-block size (i.e. N/2, the positive spectrum)
        # and zero pad it as needed.
        if not samples % mdct_lines:
            samples += (mdct_lines - samples % mdct_lines)

        # Add in another delay block for the overlap-and-add reconstruction
        samples += mdct_lines

        # Write out the file attributes
        header = struct.pack('<LHLLHH', sample_rate, channels, samples,
                             mdct_lines, scale_bits, mant_bits)
        self.file.write(header)

        # Write out to the file the scale factor allocations
        alloc = AssignMDCTLinesFromFreqLimits(mdct_lines, sample_rate)
        band_scale_factors = ScaleFactorBands(alloc)
        band_coding = struct.pack('<L', band_scale_factors.nBands)
        line_coding = struct.pack('<' + str(band_scale_factors.nBands) + 'H',
                                  *(band_scale_factors.nLines.tolist()))
        self.file.write(band_coding)
        self.file.write(line_coding)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.file.close()

    def write_data(self, block_size, block_data):
        """ Write a block of signed float data in the range of -1...1 to the
        PACfile the object holds a reference to.
        """
        if len(block_size) != len(block_data):
            raise ValueError('block_size and block_data length much be equal.')

        for (size, data) in zip(block_size, block_data):
            self.file.write(struct.pack('<L', int(size)))
            self.file.write(data)


class PACReader(object):

    def __init__(self, filename):
        """ Provide the filename of the PAC file to read in. The file is opened
        and header read upon intialization.
        """
        self.file = open(filename, 'rb')

        # Ensure we are reading PAC files
        tag = self.file.read(4)
        if tag != 'PAC ':
            raise IOError('Provided a non-PAC file to read.')

        # Grab all the delicious metadata
        (a, b, c, d, e, f) = struct.unpack('<LHLLHH', self.file.read(struct.calcsize('<LHLLHH')))
        self.sample_rate = a
        self.channels = b
        self.mdct_lines = d
        self.samples_per_block = d
        self.scale_bits = e
        self.mant_bits = f

        # Compute the scale factor bands using the delicious metadata
        bands = struct.unpack('<L', self.file.read(struct.calcsize('<L')))[0]
        lines = struct.unpack('<' + str(bands) + 'H',
                              self.file.read(struct.calcsize('<' + str(bands) + 'H')))
        self.band_scale_factors = ScaleFactorBands(lines)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.file.close()

    def metadata(self):
        return {'sample_rate': self.sample_rate, 
                'channels': self.channels,
                'mdct_lines': self.mdct_lines,
                'samples_per_block': self.samples_per_block,
                'scale_bits': self.scale_bits,
                'mant_bits': self.mant_bits}
 
    def read_block(self):
        """ Read a block of data from the PAC file. """
        block = []
        for ch in range(self.channels):
            # Figure out how much data is in this block
            word = self.file.read(struct.calcsize('<L'))

            # Check if at the last block and need to finish the overlap-and-add
            # Do this by looking if there is a header to the frame telling us
            # how much data needs to be read from the block.
            if not word:
                return None, 0

            # Otherwise, if there is data, return the size and the data block
            bytes = struct.unpack('<L', word)[0]
            pb = PackedBits()
            pb.SetPackedData(self.file.read(bytes))

            # Make sure we have a full block and not some corrupted file
            if pb.nBytes < bytes:
                raise IOError('Only read a partial block of data.')
            
            # Read information at the top of data block
            window_state = pb.ReadBits(2) 
            overall_scale = pb.ReadBits(self.scale_bits)

            # From the window state, we need to recompute the how the MDCT
            # lines were allocated across the critical bands for this block
            if window_state == 0:
                self.mdct_lines = 512
            elif window_state == 1:
                self.mdct_lines = (512 + 64) / 2
            elif window_state == 2:
                self.mdct_lines = 64
            elif window_state == 3:
                self.mdct_lines = (512 + 64) / 2
            else:
                raise ValueError('Invalid window state: ' + str(window_state))
            alloc = AssignMDCTLinesFromFreqLimits(self.mdct_lines, self.sample_rate)
            self.band_scale_factors = ScaleFactorBands(alloc)

            # Now extract the data
            bit_alloc = []
            scale_factor = []
            mant = np.zeros(self.mdct_lines, dtype=np.int32)
            for band in range(self.band_scale_factors.nBands):
                alloc = pb.ReadBits(self.mant_bits)
                alloc += 1 if alloc else 0
                bit_alloc.append(alloc)
                scale_factor.append(pb.ReadBits(self.scale_bits))

                # If there are bits allocated to this band, extract the
                # mantiassas and place them into the mantissa array
                if bit_alloc[band]:
                    m = np.empty(self.band_scale_factors.nLines[band], dtype=np.int32)
                    for i in range(self.band_scale_factors.nLines[band]):
                        m[i] = pb.ReadBits(bit_alloc[band])
                    lower = self.band_scale_factors.lowerLine[band]
                    upper = self.band_scale_factors.upperLine[band] + 1
                    mant[lower:upper] = m

            # Return what we read from the block
            block.append({'scale_factor': scale_factor,
                          'bit_alloc': bit_alloc,
                          'mantissa': mant,
                          'overall_scale': overall_scale})

        return block, window_state
