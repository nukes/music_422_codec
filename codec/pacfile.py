'''
Class to wrap the PAC file as specified by Drs. Bosi and Goldberg.
'''

import struct

import numpy as np

from provided.bitpack import PackedBits
from codec.coder import encode, decode
from codec.psychoac import ScaleFactorBands, AssignMDCTLinesFromFreqLimits


class PACReader(object):

    def __init__(self, filename):
        ''' Provide the filename of the PAC file to read in. The file is opened
        and header read upon intialization.
        '''
        self.file = open(filename, 'rb')
        self.tag = 'PAC '

        # Ensure we are reading PAC files
        tag = self.file.read(4)
        if tag != self.tag:
            raise IOError('Provided a non-PAC file to read.')

        # Grab all the delicious metadata
        (a, b, c, d, e, f) = struct.unpack('<LHLLHH',
                                 self.file.read(struct.calcsize('<LHLLHH')))
        self.sample_rate = a
        self.channels = b
        self.mdct_lines = d
        self.samples_per_block = d
        self.scale_bits = e
        self.mant_bits = f

        # Compute the scale factor bands
        bands = struct.unpack('<L', self.file.read(struct.calcsize('<L')))[0]
        lines = struct.unpack('<' + str(bands) + 'H',
                              self.file.read(struct.calcsize('<' + str(bands) + 'H')))
        self.band_scale_factors = ScaleFactorBands(lines)

        # Prime the previous block needed for the overlap-and-add
        prev_block = []
        for ch in range(self.channels):
            prev_block.append(np.zeros(self.mdct_lines, dtype=np.float64))
        self.overlap_block = prev_block
 
    def read_block(self):
        ''' Read a block of data from the PAC file. '''

        data = []
        for ch in range(self.channels):

            # Figure out how much data is in this block
            # TODO: This is where we would catch different window sizes in block-switching
            data.append(np.array([], dtype=np.float64))
            word = self.file.read(struct.calcsize('<L'))

            # Check if at the last block and need to finish the overlap-and-add
            # Do this by looking if there is a header to the frame telling us
            # how much data needs to be read from the block.
            if not word and self.overlap_block:
                self.overlap_block = None
                # TOOD: Fix by adding a "previous window" setting
                return self.overlap_block, 0
            if not word and not self.overlap_block:
                return None, None

            # Compute the number of bytes we need to read
            bytes = struct.unpack('<L', word)[0]
            pb = PackedBits()
            pb.SetPackedData(self.file.read(bytes))

            # Make sure we have a full block and not some corrupted file
            if pb.nBytes < bytes:
                raise IOError('Only read a partial block of data.')

            # Read information at the top of data block
            win_state = pb.ReadBits(2) 
            overall_scale = pb.ReadBits(self.scale_bits)

            # From the window state, we need to recompute the how the MDCT
            # lines were allocated across the critical bands for this block
            if win_state == 0:
                boundary = 512
                self.mdct_lines = 512
            elif win_state == 1:
                boundary = 512
                self.mdct_lines = (512 + 64) / 2
            elif win_state == 2:
                boundary = 64
                self.mdct_lines = 64
            elif win_state == 3:
                boundary = 64
                self.mdct_lines = (512 + 64) / 2
            else:
                raise ValueError('Invalid window state: ' + str(win_state))
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

            # Decode the data
            decoded = decode(scale_factor, bit_alloc, mant, overall_scale,
                             self.mdct_lines, self.scale_bits, 
                             self.band_scale_factors, win_state)
            data[ch] = np.concatenate([data[ch], np.add(self.overlap_block[ch], decoded[:boundary])])
            self.overlap_block[ch] = decoded[boundary:]

        return data, win_state

    def close(self):
        self.file.close()


class PACWriter(object):

    def __init__(self, filename, sample_rate, channels, samples, mdct_lines, scale_bits, mant_bits, target_bps):
        ''' Provide parameters to fully specify a PAC file. The file is opened
        and the header is written upon initialization. '''
        self.file = open(filename, 'wb')
        self.tag = 'PAC '
        self.file.write(self.tag)

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
        self.band_scale_factors = ScaleFactorBands(alloc)
        band_coding = struct.pack('<L', self.band_scale_factors.nBands)
        line_coding = struct.pack('<' + str(self.band_scale_factors.nBands) + 'H',
                                  *(self.band_scale_factors.nLines.tolist()))
        self.file.write(band_coding)
        self.file.write(line_coding)

        # Start with the first block of data for the MDCT block
        self.previous_block = []
        for ch in range(channels):
            self.previous_block.append(np.zeros(mdct_lines, dtype=np.float64))

        # Save the rest of the parameters so we can use them when writing
        self.sample_rate = sample_rate
        self.channels = channels
        self.samples = samples
        self.mdct_lines = mdct_lines
        self.scale_bits = scale_bits
        self.mant_bits = mant_bits
        self.target_bps = target_bps
        self.bitReservoir = 0

    def write_data(self, data, win_state):
        ''' Write a block of signed float data in the range of -1...1 to the
        PACfile the object holds a reference to.
        '''

        # Alter the mdct_line attribute and alter the concat block size
        # based on the window state.
        if win_state == 0:
            boundary = 512
            self.mdct_lines = 512
        elif win_state == 1:
            boundary = 64
            self.mdct_lines = (512 + 64) / 2
        elif win_state == 2:
            boundary = 64
            self.mdct_lines = 64
        elif win_state == 3:
            boundary = 512
            self.mdct_lines = (512 + 64) / 2
        else:
            raise ValueError('Invalid window state: ' + str(win_state))
        alloc = AssignMDCTLinesFromFreqLimits(self.mdct_lines, self.sample_rate)
        self.band_scale_factors = ScaleFactorBands(alloc)

        # Construct the full block to write out
        # We need to use the previous block for the overlap-and-add requirement
        # Removed the 'boundary' from the data slice
        full_block = []
        for ch in range(self.channels):
            block = np.concatenate([self.previous_block[ch], data[ch]])
            full_block.append(block)
            self.previous_block[ch] = data[ch]

        (scale_factor, bit_alloc, mant, overall_scale, rem_bits) = encode(full_block,
                                                                win_state,
                                                                self.channels,
                                                                self.sample_rate,
                                                                self.mdct_lines,
                                                                self.scale_bits,
                                                                self.mant_bits,
                                                                self.band_scale_factors,
                                                                self.target_bps,
                                                                self.bitReservoir)

        # Update bit reservoir for next data block
        if self.bitReservoir < 10**17:
            self.bitReservoir += rem_bits[0] #
        #print self.bitReservoir

        # Write the encoded data to the file
        for ch in range(self.channels):

            # Determine the size of the channel's block
            bits = self.scale_bits
            for band in range(self.band_scale_factors.nBands):
                bits += self.mant_bits + self.scale_bits
                if bit_alloc[ch][band]:
                    bits += bit_alloc[ch][band] * self.band_scale_factors.nLines[band]

            # TODO: This is where we count the bits needed for block switching
            # i.e. This is where we add '2' to the count of bits needed
            # Add in the two bits to encode what kind of window this would be
            bits += 2

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
            # First we will pack the window information, then everything else!
            pb.WriteBits(win_state, 2)
            pb.WriteBits(overall_scale[ch], self.scale_bits)
            i_mant = 0
            for band in range(self.band_scale_factors.nBands):
                alloc = bit_alloc[ch][band]
                if alloc:
                    alloc -= 1
                pb.WriteBits(alloc, self.mant_bits)
                pb.WriteBits(scale_factor[ch][band], self.scale_bits)
                if bit_alloc[ch][band]:
                    for i in range(self.band_scale_factors.nLines[band]):
                        pb.WriteBits(mant[ch][i_mant+i], bit_alloc[ch][band])
                    i_mant += self.band_scale_factors.nLines[band]

            # Finally, write this damned data to the file
            self.file.write(pb.GetPackedData())

    def close(self):
        self.file.close()
