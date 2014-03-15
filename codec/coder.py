""" The coder pipeline """

import numpy as np

from codec.quantize import ScaleFactor, vMantissa, vDequantize
from codec.window import compose_kbd_window
from codec.mdct import MDCT, IMDCT
from codec.psychoac import CalcSMRs
from codec.bitalloc import BitAlloc


def decode(scale_factor, bit_alloc, mant, overall_scale, mdct_lines, scale_bits, band_scale_factors, window_state):
    """ Decode the multi-channel data stream """

    half_long = 512
    half_short = 64

    # Dequantize the block
    mdct_data = np.zeros(mdct_lines, dtype=np.float64)
    mant_count = 0
    for band in range(band_scale_factors.nBands):
        lines = band_scale_factors.nLines[band]
        if bit_alloc[band]:
            mdct_data[mant_count:mant_count+lines] = vDequantize(scale_factor[band],
                                                                 mant[mant_count:mant_count+lines],
                                                                 scale_bits,
                                                                 bit_alloc[band])
        mant_count += lines


    # Scale the data back by the overall gain we original gave the block
    mdct_data = mdct_data / float(1 << overall_scale)

    # Perform the correct inverse MDCT function based on the window of the block
    # This will give us perfect reconstruction in the synthesis step
    if window_state == 0:
        half_n = half_long
        samples = IMDCT(mdct_data, half_long, half_long)
        win_samples = compose_kbd_window(samples, half_long, half_long, 4., 4.)
        #win_samples = SineWindow(samples)
    elif window_state == 1:
        half_n = (half_long + half_short) / 2
        samples = IMDCT(mdct_data, half_long, half_short)
        win_samples = compose_kbd_window(samples, half_long, half_short, 4., 4.)
    elif window_state == 2:
        half_n = half_short
        samples = IMDCT(mdct_data, half_short, half_short)
        win_samples = compose_kbd_window(samples, half_short, half_short, 4., 4.)
    elif window_state == 3:
        half_n = (half_long + half_short) / 2
        samples = IMDCT(mdct_data, half_short, half_long)
        win_samples = compose_kbd_window(samples, half_short, half_long, 4., 4.)
    else:
        raise ValueError('Unknown window state:' + str(window_state))

    return win_samples


def encode(data, window_state, channels, sample_rate, mdct_lines, scale_bits, mant_bits, band_scale_factors, target_bps, bitReservoir):
    """ Encode the multi-channel data stream """

    scale_factors = []
    bit_alloc = []
    mant = []
    overall_scale = []
    rem_bits = []

    for data_channel in data:
        (a, b, c, d, e) = encode_channel(data_channel,
                                      window_state,
                                      channels, 
                                      sample_rate,
                                      mdct_lines,
                                      scale_bits,
                                      mant_bits,
                                      band_scale_factors,
                                      target_bps,
                                      bitReservoir);
        scale_factors.append(a)
        bit_alloc.append(b)
        mant.append(c)
        overall_scale.append(d)
        rem_bits.append(e)

    return (scale_factors, bit_alloc, mant, overall_scale, rem_bits)


def encode_channel(samples, window_state, channels, sample_rate, mdct_lines, scale_bits, mant_bits, band_scale_factors, target_bps, bitReservoir):
    """ Encode a single channel of data through the pipeline. Each channel,
    in this coder, is just considered in isolation -- i.e. we have n channels
    of mono audio, we consider no correlation between them.
    """

    # Constants that are use throughout the encoder chain. These are defined by
    # the block switching window sizes. The also represent the MDCT lines
    # available to the coder for that window size.
    half_long = 512
    half_short = 64
    max_mant_bits = 16 if mant_bits > 16 else (1 << mant_bits)

    # Perform the MDCT and set the true MDCT count based on which window we are
    # in. Additionally, tune the KBD alpha for frequency selectivity.
    if window_state == 0:
        half_n = half_long
        win_samples = compose_kbd_window(samples, half_long, half_long, 4., 4.)
        mdct_data = MDCT(win_samples, half_long, half_long)[:half_n]
    elif window_state == 1:
        half_n = (half_long + half_short) / 2
        win_samples = compose_kbd_window(samples, half_long, half_short, 4., 6.)
        mdct_data = MDCT(win_samples, half_long, half_short)[:half_n]
    elif window_state == 2:
        half_n = half_short
        win_samples = compose_kbd_window(samples, half_short, half_short, 6., 6.)
        mdct_data = MDCT(win_samples, half_short, half_short)[:half_n]
    elif window_state == 3:
        half_n = (half_long + half_short) / 2
        win_samples = compose_kbd_window(samples, half_short, half_long, 6., 4.)
        mdct_data = MDCT(win_samples, half_short, half_long)[:half_n]
    else:
        raise ValueError('Unknown window state:' + str(window_state))

    # Compute the overall scale factor for this block and apply the gain to
    # the MDCT data we have right now.
    max_line = np.max(np.abs(mdct_data))
    overall_scale = ScaleFactor(max_line, scale_bits)
    mdct_data *= (1 << overall_scale)

    # !!! Psychoacoustic side chain !!!
    # Compute the signal-to-mask ratio
    smr_data = CalcSMRs(samples, mdct_data, overall_scale, sample_rate, band_scale_factors)
    smr_data = np.array(smr_data)

    # Compute the bit budget for this block given the target bps rate we want
    # to achieve. Also, account for the two bits we will lose due to block
    # switching so that we remain critically sampled. Finally, if we are in a
    # short window, up the data rate we so don't have frequency thrashing.
    budget = target_bps * half_n
    budget += bitReservoir if window_state == 2 else 0
    budget -= scale_bits * (band_scale_factors.nBands + 1)
    budget -= mant_bits * band_scale_factors.nBands
    budget -= 2
    budget = int(np.floor(budget))

    # Figure out how to allocate the bit budget give the signal-to-mask
    # perceptual indicators. Larger SMR values in a critical band mean that the
    # frequencies within that band require more bits to express. We are
    # reducing quantization noise by allocation more bits where we hear content
    (bit_alloc, remBits) = BitAlloc(budget, max_mant_bits, band_scale_factors.nBands, band_scale_factors.nLines, smr_data)

    # Using these bit allocations, quantize the MDCT data for each band using
    # the perceptual bit resolution. First, figure out how many bits we are
    # going to need for the mantissa.
    scale_factor = np.empty(band_scale_factors.nBands, dtype=np.int32)
    total_mant_bits = half_n
    for band in range(band_scale_factors.nBands):
        if not bit_alloc[band]:
            total_mant_bits -= band_scale_factors.nLines[band]

    # Now, allocate the bits for each band
    mant = np.empty(total_mant_bits, dtype=np.int32)
    mant_count = 0
    for band in range(band_scale_factors.nBands):
        lower = band_scale_factors.lowerLine[band]
        lines = band_scale_factors.nLines[band]
        upper = lower + lines

        # --- Still a weird patch to apply...
        if lower >= upper:
            scale_line = 1
        else:
            #print "defect"
            scale_line = np.max(np.abs(mdct_data[lower:upper]))
        # --- end the weird patch 

        scale_factor[band] = ScaleFactor(scale_line, scale_bits, bit_alloc[band])
        if bit_alloc[band]:
            mant[mant_count:mant_count+lines] = vMantissa(mdct_data[lower:upper],
                                                          scale_factor[band],
                                                          scale_bits,
                                                          bit_alloc[band])
            mant_count += lines

    return (scale_factor, bit_alloc, mant, overall_scale, remBits)
