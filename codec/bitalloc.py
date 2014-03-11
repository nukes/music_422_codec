import numpy as np


DBTOBITS = 6.02
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
           We will not bother to worry about slight variations in bit budget due to
           rounding of the above equation to integer values of R(i).
      """

    # Enforce 16 bit input
    if maxMantBits > 16: 
        maxMantBits = 16

    # Initialize vector containing number of bits allocated to each band
    bits = np.zeros(nBands, dtype=np.int32)
    remBits = bitBudget    # remaining bits in budget
    lastBudget = bitBudget    # remaining bit budget from last iteration

    # Flags for which bands we can allocate to
    # Initialize it such that everything can be allocated to
    availableBands = np.ones(nBands, dtype=bool)

    # Water-filling bit allocation
    while True:

        # Are there no more bands to fill anymore? If so, break the rate
        # distortion loop.
        if not np.any(availableBands):
            break

        # Indices of bands we're not done filling (the first max value in each available band)
        #print (SMR < max(SMR[availableBands])).nonzero()[0]
        indices = (SMR == max(SMR[availableBands])).nonzero()[0]
        
        if indices.size == 0: break

        for i in indices:
          #print i, remBits, nLines[i]
          if remBits >= nLines[i]:
            if bits[i] < maxMantBits:
              remBits -= nLines[i]
              bits[i] += 1
            elif bits[i] == maxMantBits:
              availableBands[i] = False
            SMR[i] -= DBTOBITS
          else:
            availableBands[i] = False
        
        # Stop if no bits were assigned in the last iteration
        if remBits == lastBudget: break
        lastBudget = remBits

    # Check for negative values and remove them
    bits[bits<1] = 0

    # Pair up any leftover 1 bit allocations
    bits = _pair_ones(bits)

    # Remove a lone 1 bit that couldn't be paired, if it exists
    bits[bits==1] = 0

    return bits, remBits


def _pair_ones(alloc):

  while True:
      ones = np.where(alloc==1)[0]
      if len(ones) == 0:
          return alloc
      high = np.max(ones)
      low = np.min(ones)
      if high == low:
          return alloc
      alloc[low] += 1
      alloc[high] -= 1
