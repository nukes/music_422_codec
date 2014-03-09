"""
quantize.py -- routines to quantize and dequantize floating point values
between -1.0 and 1.0 ("signed fractions")
"""

### ADD YOUR CODE AT THE SPECIFIED LOCATIONS ###

import numpy as np

### Problem 1.a.i ###
def QuantizeUniform(aNum,nBits):
    """
    Uniformly quantize signed fraction aNum with nBits
    """
    #Notes:
    #The overload level of the quantizer should be 1.0

    ### YOUR CODE STARTS HERE ###
    s = 0
    if aNum < 0:
        s = 1

    aNum = abs(aNum)
    code = 0
    if aNum >= 1:
        code = 2**(nBits - 1) - 1
    else:
        code = int(((2**nBits - 1)*aNum + 1)/2)

    aQuantizedNum = (s << (nBits - 1)) | code
    
    ### YOUR CODE ENDS HERE ###

    return aQuantizedNum

### Problem 1.a.i ###
def DequantizeUniform(aQuantizedNum,nBits):
    """
    Uniformly dequantizes nBits-long number aQuantizedNum into a signed fraction
    """
    
    ### YOUR CODE STARTS HERE ###

    signMask = 1 << (nBits - 1)
    s = (aQuantizedNum & signMask) >> (nBits - 1)
    if s == 0:
        sign = 1
    else:
        sign = -1

    aNum = sign * (2* float(aQuantizedNum & (~signMask))/(2**nBits - 1))
    
    ### YOUR CODE ENDS HERE ###

    return aNum

### Problem 1.a.ii ###
def vQuantizeUniform(aNumVec, nBits):
    """
    Uniformly quantize vector aNumberVec of signed fractions with nBits
    """
    
    ### YOUR CODE STARTS HERE ###

    if nBits <= 0:
        return np.zeros(len(aNumVec),dtype=np.uint64)
    signBit = (1<<(nBits-1))
    multFac = (signBit<<1) - 1
    val = aNumVec.copy()
    isign = np.signbit(val)
    val[isign]=-val[isign]
    code = np.empty(len(val),dtype=np.uint64)

    code[val>=1]=signBit-1
    code[val!=1]=((val[val!=1]*multFac + 1.)/2.).astype(np.uint64)
    code[isign] += signBit
    
    ### YOUR CODE ENDS HERE ###

    return code

### Problem 1.a.ii ###
def vDequantizeUniform(aQuantizedNumVec, nBits):
    """
    Uniformly dequantizes vector of nBits-long numbers aQuantizedNumVec into vector of  signed fractions
    """
    
    ### YOUR CODE STARTS HERE ###

    if nBits <= 0:
        return np.zeros(len(aQuantizedNumVec),dtype=np.float64)
    signBit = (1<<(nBits-1))
    multFac = (signBit << 1)-1
    code = aQuantizedNumVec.copy()

    sign = np.zeros(len(aQuantizedNumVec),dtype=np.bool)
    negs = (aQuantizedNumVec & signBit) == signBit
    sign[negs] = True
    code[negs] -= signBit
    val = 2.*code/multFac
    val[sign] = -val[sign]
    
    ### YOUR CODE ENDS HERE ###

    return val

### Problem 1.b ###
def ScaleFactor(aNum, nScaleBits=3, nMantBits=5):
    """
    Return the floating-point scale factor for a  signed fraction aNum given nScaleBits scale bits and nMantBits mantissa bits
    """
    #Notes:
    #The scale factor should be the number of leading zeros
    
    ### YOUR CODE STARTS HERE ###
    if nScaleBits < 0: nScaleBits = 0
    if nMantBits <= 0: return 0

    maxScale = (1<<nScaleBits)-1
    maxBits = maxScale + nMantBits
    signBit = (1<<(maxBits-1))
    code = QuantizeUniform(abs(aNum),maxBits)
    code <<= 1
    scale = 0
    while scale < maxScale and (signBit & code) == 0:
        code <<= 1
        scale += 1
        
    ### YOUR CODE ENDS HERE ###

    return scale

### Problem 1.b ###
def MantissaFP(aNum, scale, nScaleBits=3, nMantBits=5):
    """
    Return the floating-point mantissa for a  signed fraction aNum given nScaleBits scale bits and nMantBits mantissa bits
    """

    ### YOUR CODE STARTS HERE ###

    if nMantBits <= 0: return 0
    if nScaleBits < 0: nScaleBits = 0

    maxScale = (1<<nScaleBits)-1
    maxBits = maxScale + nMantBits
    signBit = (1<<(nMantBits-1))

    sign = 0
    if aNum < 0:
        sign = 1
        aNum *= -1
    code = QuantizeUniform(aNumber, maxBits)
    code <<= (scale+1)
    if scale < maxScale:
        code -= (1<<(maxBits-1))
        code <<= 1
    code >>= (maxBits - nMantBits + 1)
    if sign: code += signBit
                
    ### YOUR CODE ENDS HERE ###

    return code

### Problem 1.b ###
def DequantizeFP(scale, mantissa, nScaleBits=3, nMantBits=5):
    """
    Returns a  signed fraction for floating-point scale and mantissa given specified scale and mantissa bits
    """

    ### YOUR CODE STARTS HERE ###

    if nMantBits <= 0: return 0
    if nScaleBits < 0: nScaleBits = 0

    maxScale = (1<<nScaleBits)-1
    maxBits = maxScale + nMantBits
    signBit = (1<<(nMantBits - 1))

    if mantissa & signBit:
        sign = 1
        mantissa -= signBit
    else: sign = 0

    if scale < maxScale:
        mantissa = mantissa + (1<<(nMantBits-1))

    if scale < (maxScale -1 ):
        mantissa = (mantissa <<1) + 1
        mantissa <<= (maxScale - scale - 2)

    if sign:
        signBit = (1<<(maxBits -1))
        mantissa += signBit
    
    ### YOUR CODE ENDS HERE ###

    return DequantizeUniform(mantissa, maxBits)

### Problem 1.c.i ###
def Mantissa(aNum, scale, nScaleBits=3, nMantBits=5):
    """
    Return the block floating-point mantissa for a  signed fraction aNum given nScaleBits scale bits and nMantBits mantissa bits
    """

    ### YOUR CODE STARTS HERE ###
    
    if nMantBits <= 0: return 0
    if nScaleBits < 0: nScaleBits = 0

    maxScale = (1 << nScaleBits) - 1
    maxBits = maxScale + nMantBits
    signBit = (1<<(nMantBits - 1))

    sign = 0
    if aNum < 0:
        sign = 1
        aNum *= -1

    code = QuantizeUniform(aNum, maxBits)
    code <<= 1
    code <<= scale
    code >>= (maxBits - nMantBits + 1)

    if sign: code += signBit
    
    ### YOUR CODE ENDS HERE ###

    return code

### Problem 1.c.i ###
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

### Problem 1.c.ii ###
def vMantissa(aNumVec, scale, nScaleBits=3, nMantBits=5):
    """
    Return a vector of block floating-point mantissas for a vector of  signed fractions aNum given nScaleBits scale bits and nMantBits mantissa bits
    """

    ### YOUR CODE STARTS HERE ###

    if nMantBits <= 0: return np.zero(len(aNumVec), np.uint64)
    if nScaleBits < 0: nScaleBits = 0

    maxScale = (1<<nScaleBits) - 1
    maxBits = maxScale + nMantBits
    signBit = (1<<(nMantBits - 1))

    val = aNumVec.copy()
    sign = np.signbit(val)
    val[sign] = -val[sign]

    code = vQuantizeUniform(val,maxBits)
    code <<= (scale + 1)
    code >>= (maxBits - nMantBits + 1)
    code[sign] += signBit
        
    ### YOUR CODE ENDS HERE ###

    return code

### Problem 1.c.ii ###
def vDequantize(scale, mantissaVec, nScaleBits=3, nMantBits=5):
    """
    Returns a vector of  signed fractions for block floating-point scale and vector of block floating-point mantissas given specified scale and mantissa bits
    """

    ### YOUR CODE STARTS HERE ###

    if nMantBits <= 0: return np.zero(len(aNumVec), np.uint64)
    if nScaleBits < 0: nScaleBits = 0

    maxScale = (1<<nScaleBits) - 1
    maxBits = maxScale + nMantBits
    signBit = (1<<(nMantBits - 1))
    mantissa = mantissaVec.copy()
    negs = (mantissa & signBit) == signBit
    mantissa[negs] -= signBit

    code = mantissa << (maxScale - scale)
    if scale < maxScale: code[mantissa > 0] += 1 << (maxScale - scale - 1)
    signBit = (1 << (maxBits - 1))
    code[negs] += signBit

    ### YOUR CODE ENDS HERE ###
    
    return vDequantizeUniform(code, maxBits)

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":

    ### YOUR TESTING CODE STARTS HERE ###

    nb = 12
    x = QuantizeUniform(-0.41, nb)

    v = np.array([-1.0,-.98,-0.41,-0.02,0.0,0.01,0.35,0.78,0.99,1.0])
    xx = vQuantizeUniform(v,nb)
    #print('Vector Uniform quantized: %s' % xx)
    #print('Vector Uniform dequantized: %s' % vDequantizeUniform(xx,nb))

    nb = 12
    xx = vQuantizeUniform(v,nb)
    #print('Vector Uniform quantized: %s' % xx)
    #print('Vector Uniform dequantized: %s' % vDequantizeUniform(xx,nb))

    x = -0.41
    scale = ScaleFactor(x)
    #print("Scale: ", scale)
    mantissa = MantissaFP(x, scale)
    #print("MantissaFP: ", mantissa)
    aNum = DequantizeFP(scale, mantissa)
    #print("DequantizedFP: ", aNum)

    mantissa = Mantissa(x, scale)
    #print("Block mantissa: ", mantissa)
    aNum = Dequantize(scale, mantissa)
    #print("Dequantized block: ", aNum)

    scale = ScaleFactor(np.max(np.absolute(v)))
    mvec = vMantissa(v, scale)
    print("Mantissa BFP quantized: %s" % mvec)
    out = vDequantize(scale, mvec)
    print("Vector BFP dequantized: %s" % out)
    
    ### YOUR TESTING CODE ENDS HERE ###

