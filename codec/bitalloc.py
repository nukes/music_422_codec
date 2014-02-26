import numpy as np


def water_fill(signal, bit_alloc, bits_remaining, max_mant, nLines):

    signal = np.array(signal)
    threshold = np.max(0.5 * np.log2(signal ** 2))
    transform = 0.5 * np.log2(signal ** 2)

    sorted_transform = np.sort(transform)
    indices = np.argsort(transform)

    while bits_remaining > np.min(nLines):

        threshold -= 1

        for i in range(len(sorted_transform)):
            k = indices[i]
            bit_fill = nLines[k]
            if sorted_transform[k] > threshold and bit_fill <= bits_remaining \
                                               and bit_alloc[k] < max_mant: 
                bit_alloc[k] += 1
                bits_remaining -= bit_fill

    # Get rid of ones!
    bits_without_ones = np.where(bit_alloc == 1, 0, bit_alloc)
    ones_remaining = np.sum(bit_alloc) - np.sum(bits_without_ones)

    # Iteratively dole out the one bits
    for i in range(int(ones_remaining)):
        k = np.argsort(bits_without_ones[bits_without_ones >= 2])[0]
        bits_without_ones[k] += 1

    return bits_without_ones.astype(int)


def BitAlloc(bitBudget, maxMantBits, nBands, nLines, SMR):
    """
    Allocates bits to scale factor bands so as to flatten the NMR across the spectrum
    
       Arguments:
           bitBudget is total number of mantissa bits to allocate
           maxMantBits is max mantissa bits that can be allocated per line
           nBands is total number of scale factor bands
           nLines[nBands] is number of lines in each scale factor band
           SMR[nBands] is signal-to-mask ratio in each scale factor band
    
        Return:
            bits[nBands] is number of bits allocated to each scale factor band
    
        Logic:
           Maximizing SMR over block gives optimization result that:
               R(i) = P/N + (1 bit/ 6 dB) * (SMR[i] - avgSMR)
           where P is the pool of bits for mantissas and N is number of bands
           This result needs to be adjusted if any R(i) goes below 2 (in which
           case we set R(i)=0) or if any R(i) goes above maxMantBits (in
           which case we set R(i)=maxMantBits).  (Note: 1 Mantissa bit is
           equivalent to 0 mantissa bits when you are using a midtread quantizer.)
           We will not bother to worry about slight variations in bit budget due
           rounding of the above equation to integer values of R(i).
      """


    """
    uniform = np.int32([ 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    constSNR = np.int32([ 2, 2, 2, 4, 4, 4, 8, 12, 9, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    constMNR = np.int32([ 3, 3, 2, 0, 3, 0, 0, 3, 2, 2, 3, 3, 3, 3, 3, 3, 0, 2, 3, 3, 3, 3, 4, 4, 5])

    uniform = np.int32([ 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
    constSNR = np.int32([ 8, 8, 8, 12, 9, 10, 16, 16, 16, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    constSMR = np.int32([ 6, 6, 5, 3, 6, 0, 3, 6, 5, 5, 6, 6, 6, 6, 6, 6, 4, 4, 5, 5, 6, 5, 7, 7, 8])


    #return uniform
    return constSNR
    #return constMNR
    """
    
    Kp = len(SMR[SMR > 0])
    Rk = np.zeros(nBands, dtype=np.int32)

    threshold = np.ceil(0.5*np.log2(max(np.round(SMR)**2)))
    cutSMR = np.copy(np.round(SMR))
    cutSMR[cutSMR <= 2] = 3
    cutSMR[SMR<-20] = 1

    v = np.round(0.5*np.log2(cutSMR**2))

    v = np.abs(v)

    # water-filling
    remBits = bitBudget
    while threshold > 0:
      Rk[ v >= threshold ] += 1
      threshold -= 1
      remBits -= len(Rk[ v >= threshold ])
      if remBits <= 0: break

    # check for single bits and negative values
    ones = len(Rk[Rk==1])
    Rk[Rk<=1] = 0
    
    # redistribute isolated bits
    while ones > 0:
      for i in range(0,len(Rk)):
        if Rk[i] > 1:
          Rk[i] += 1
          ones -= 1

    for i in range(0,len(Rk)):
      if remBits >= 3:
        Rk[Rk>0] += 3
    # check for bit overflow
    Rk[Rk>maxMantBits] = maxMantBits

    return Rk

