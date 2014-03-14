'''
Block floating-point quantization routines.
All input is expected to be normalized between 1.0 and -1.0

Encode a block using ScaleFactor and vMantissa, and decode with vDequantize.
'''
import numpy as np


def ScaleFactor(sample, scale_bits=3, mant_bits=5):
    ''' Return the scale factor for a signed fraction in block FP scheme '''
    # Guard against bad and degenerate conditions
    if mant_bits <= 0:
        return 0
    scale_bits = max(scale_bits, 0)

    max_scale, max_bits = _get_max_scale_size(scale_bits, mant_bits)
    sign_bit = (1 << (max_bits - 1))

    code = QuantizeUniform(abs(sample), max_bits) << 1
    scale = 0
    while scale < max_scale and (sign_bit & code) == 0:
        code <<= 1
        scale += 1

    return scale


def vMantissa(samples, scale, scale_bits=3, mant_bits=5):
    ''' Return a vector of mantissas in block FP for a given sample vector '''
    # Guard against bad and degenerate conditions
    if mant_bits <= 0:
        return np.zeros(len(samples)).astype(int)
    scale_bits = max(scale_bits, 0)

    max_bits = (1 << scale_bits) - 1 + mant_bits
    sign_bit = (1 << (mant_bits - 1))

    val = samples.copy()
    sign_mask = np.signbit(val)
    val[sign_mask] = -val[sign_mask]

    code = vQuantizeUniform(val, max_bits)
    code <<= (scale + 1)
    code >>= (max_bits - mant_bits + 1)
    code[sign_mask] += sign_bit

    return code


def vDequantize(scale, mantissa, scale_bits=3, mant_bits=5):
    ''' Return the dequantized, signed fraction values from the given
    block scale factor and mantissa vector, using the block FP scheme '''
    # Guard against bad and degenerate conditions
    if mant_bits <= 0:
        return 0.
    scale_bits = max(scale_bits, 0)
    mant = mantissa.copy()

    max_scale, max_bits = _get_max_scale_size(scale_bits, mant_bits)
    sign_bit = 1 << (mant_bits - 1)

    negatives = (mant & sign_bit) == sign_bit
    mant[negatives] -= sign_bit

    ret_code = mant << (max_scale - scale)
    if scale < max_scale:
        ret_code[mant > 0] += 1 << (max_scale - scale - 1)
    sign_bit = 1 << (max_bits - 1)
    ret_code[negatives] += sign_bit

    return vDequantizeUniform(ret_code, max_bits)


def vQuantizeUniform(aNumVec, nBits):
    ''' Uniformly quantize vector aNumberVec of signed fractions with nBits '''
    vec = aNumVec.copy()

    sign_bit = 1 << (nBits-1)
    multiple = (sign_bit << 1) - 1
    sign_mask = np.signbit(vec)
    vec[sign_mask] = -vec[sign_mask]

    ret_code = np.empty(len(vec), dtype=np.uint64)
    ret_code[vec >= 1] = sign_bit - 1
    ret_code[vec != 1] = ((vec[vec != 1]*multiple + 1.)/2.).astype(np.uint64)
    ret_code[sign_mask] += sign_bit
    return ret_code


def vDequantizeUniform(aQuantizedNumVec, nBits):
    ''' Uniformly dequantize vectors of arbitrary bits '''
    sign_vec = np.right_shift(aQuantizedNumVec, nBits-1)
    sign_vec = np.multiply(sign_vec, -2)
    sign_vec = np.add(sign_vec, 1)

    val_vec = np.bitwise_and(aQuantizedNumVec, _code_abs_mask(nBits))
    val_vec = np.multiply(val_vec, 2.0)
    val_vec = np.divide(val_vec, (2 ** nBits - 1))
    return np.multiply(sign_vec, val_vec)


def QuantizeUniform(aNum, nBits):
    ''' Uniformly quantize signed fraction aNum with nBits '''
    if aNum == 0:
        return 0
    sign = int(aNum < 0)
    if abs(aNum) >= 1:
        code = _midtread_bins(nBits) - 1
    else:
        code = int(_midtread_bins(nBits) * abs(aNum))
    return (sign << (nBits - 1)) | code


def DequantizeUniform(aQuantizedNum, nBits):
    ''' Uniformly dequantizes a codeword of nBits into a signed fraction '''
    if aQuantizedNum == 0:
        return 0.
    sign = 1 - 2 * (aQuantizedNum >> (nBits - 1))
    aQuantizedNum &= _code_abs_mask(nBits)
    n = (2.0 * aQuantizedNum) / (2 ** nBits - 1)
    return sign * abs(n)


def Dequantize(scale, mantissa, nScaleBits=3, nMantBits=5):
    """
    Returns a  signed fraction for block floating-point scale and mantissa given specified scale and mantissa bits
    """

    ### YOUR CODE STARTS HERE ###

    if nMantBits <= 0: return 0
    if nScaleBits < 0: nScaleBits = 0

    maxScale = (1 << nScaleBits) - 1
    maxBits = maxScale + nMantBits
    signBit = (1<<(nMantBits-1))

    if mantissa & signBit:
        sign = 1
        mantissa -= signBit
    else: sign = 0

    code = mantissa << (maxScale-scale)
    if (scale < maxScale and mantissa > 0): code += 1<<(maxScale - scale - 1)

    if sign:
        signBit = (1<<(maxBits - 1))
        code += signBit
    
    ### YOUR CODE ENDS HERE ###

    return DequantizeUniform(code, maxBits)


def _get_max_scale_size(scale_bits, mant_bits):
    ''' Returns a tuple of the maximum scale and the maximum bits
    used in the block FP scheme calculations '''
    max_scale = (1 << scale_bits) - 1
    max_bits = max_scale + mant_bits
    return (max_scale, max_bits)


def _midtread_bins(x):
    ''' Shortcut for the number of bins in a midtread quantizer '''
    return 2 ** (x - 1)


def _code_abs_mask(bits):
    ''' Shortcut to produce a mask to get the absolute value '''
    return 2 ** (bits - 1) - 1
