""" Encoding and decoding pipeline functions. """

import numpy as np

from codec.coder import encode, decode
from codec.psychoac import AssignMDCTLinesFromFreqLimits, ScaleFactorBands
from codec.bitpack import PackedBits

class Encoder:

    def __init__(self, sample_rate, channels, mdct_lines, scale_bits, mant_bits, target_bps):
        alloc = AssignMDCTLinesFromFreqLimits(mdct_lines, sample_rate)
        self.previous_block = []
        for ch in range(channels):
            self.previous_block.append(np.zeros(mdct_lines, dtype=np.float64))

        self.band_scale_factors = ScaleFactorBands(alloc)
        self.sample_rate = sample_rate
        self.channels = channels
        self.mdct_lines = mdct_lines
        self.scale_bits = scale_bits
        self.mant_bits = mant_bits
        self.target_bps = target_bps
        self.bitReservoir = 0

    def encode_block(self, data, window_state):
        # Alter the mdct_line attribute and alter the concat block size
        # based on the window state.
        # TODO: Pull out these magic numbers and replace with sanity.
        if window_state == 0:
            boundary = 512
            self.mdct_lines = 512
        elif window_state == 1:
            boundary = 64
            self.mdct_lines = (512 + 64) / 2
        elif window_state == 2:
            boundary = 64
            self.mdct_lines = 64
        elif window_state == 3:
            boundary = 512
            self.mdct_lines = (512 + 64) / 2
        else:
            raise ValueError('Invalid window state: ' + str(window_state))

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
                                                                          window_state,
                                                                          self.channels,
                                                                          self.sample_rate,
                                                                          self.mdct_lines,
                                                                          self.scale_bits,
                                                                          self.mant_bits,
                                                                          self.band_scale_factors,
                                                                          self.target_bps,
                                                                          self.bitReservoir)

        # Update bit reservoir for next data block. 
        # This is an ugly workaround for a little buggy bit allocation...avert
        # your eyes! Pay no attention to the man behind the curtain!
        if self.bitReservoir < 180:
            self.bitReservoir += rem_bits[0]

        # Send this data out to create a single bit block for writing
        return self._create_bit_block(window_state, scale_factor, bit_alloc, mant, overall_scale)

    def _create_bit_block(self, window_state, scale_factor, bit_alloc, mant, overall_scale):
        block_size = []
        block_data = []

        # Assemble this data into a single bit block
        for ch in range(self.channels):

            # Determine the size of the channel's block
            bits = self.scale_bits
            for band in range(self.band_scale_factors.nBands):
                bits += self.mant_bits + self.scale_bits
                if bit_alloc[ch][band]:
                    bits += bit_alloc[ch][band] * self.band_scale_factors.nLines[band]

            # This is where we count the bits needed for block switching
            # i.e. This is where we add '2' to the count of bits needed
            # Add in the two bits to encode what kind of window this would be
            bits += 2

            # Convert the bits to bytes, using the conventional definition of
            # 8 bits to 1 byte.  Add a "spillover" byte if we are just shy of
            # the 8 byte boundary -- i.e. guarantee we have the byte capacity
            bytes = bits / 8 + int(bits % 8 != 0)

            # Write out the size of the data block
            block_size.append(bytes)
            
            # TODO: Eliminate this PackedBits if possible. Seems hard, though.
            pb = PackedBits()
            pb.Size(bytes)

            # Actually pack the data
            # First we will pack the window information, then everything else!
            pb.WriteBits(window_state, 2)
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
            block_data.append(pb.GetPackedData())

        return (block_size, block_data)


class Decoder:

    def __init__(self, pac_metadata):
        self.channels = pac_metadata['channels']
        self.sample_rate = pac_metadata['sample_rate']
        self.scale_bits = pac_metadata['scale_bits']

        # Prime the previous block needed for the overlap-and-add
        self.overlap_block = []
        for ch in range(self.channels):
            self.overlap_block.append(np.zeros(pac_metadata['mdct_lines'], dtype=np.float64))

    def decode_block(self, block, window):
        wav_data = []
        for ch in range(self.channels):
            if not block:
                if self.overlap_block:
                    flush_block = self.overlap_block
                    self.overlap_block = None
                    return flush_block
                else:
                    return None 
            else:
                decoded = self._decode_channel(block[ch], window)

            # Perform the overlap-and-add
            if window in (0, 1):
                boundary = 512
            elif window in (2, 3):
                boundary = 64
            else:
                raise ValueError('Invalid window state: ' + str(window_state))
            wav_data.append(np.add(self.overlap_block[ch], decoded[:boundary]))
            self.overlap_block[ch] = decoded[boundary:]
        return wav_data

    def _decode_channel(self, block, window):
        if window == 0:
            mdct_lines = 512
        elif window == 2:
            mdct_lines = 64
        elif window in (1, 3):
            mdct_lines = (512 + 64) / 2
        else:
            raise ValueError('Invalid window state: ' + str(window))

        alloc = AssignMDCTLinesFromFreqLimits(mdct_lines, self.sample_rate)
        band_scale_factors = ScaleFactorBands(alloc)

        # Return the WAV samples from this
        return decode(block['scale_factor'],
                      block['bit_alloc'],
                      block['mantissa'],
                      block['overall_scale'],
                      mdct_lines,
                      self.scale_bits,
                      band_scale_factors,
                      window)

