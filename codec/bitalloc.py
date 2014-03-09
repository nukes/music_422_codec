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

    # Update SMR vector through each bit allocation iteration
    #SMR = np.copy(SMR)
    
    ######## Water-filling ########

    remBits = bitBudget    # remaining bits in budget
    lastBudget = bitBudget    # remaining bit budget from last iteration

    # Flags for which bands we can allocate to
    # Initialize it such that everything can be allocated to
    availableBands = np.ones(nBands, dtype=bool)

    while True:

        # Are there no more bands to fill anymore? If so, break the rate
        # distortion loop.
        if not np.any(availableBands):
            break

        # If all bands are saturated -- that is, we were bit-wealthy -- then
        # break this loop.
        # TODO: This condition can be pulled outside of the loop
        if np.all(bits == maxMantBits):
            break

        # Indices of bands we're not done filling (the first max value in each available band)
        #print (SMR < max(SMR[availableBands])).nonzero()[0]
        indices = (SMR == max(SMR[availableBands])).nonzero()[0]
        
        #print "---"
        #print SMR
        #print nLines
        #print SMR == max(SMR[availableBands])

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
        #print "Waterfilling round: ", bits
        # Stop if no bits were assigned in the last iteration
        if remBits == lastBudget: break
        lastBudget = remBits

    #print 'Finish: ', bits

    # Check for single bits and negative values
    bits[bits<=1] = 0

    availableBands = (bits != maxMantBits)
    
    while True:
      if np.all(np.logical_not(availableBands)): break
      indices = (bits==1).nonzero()[0]    # indices of bands with just 1 allocated bit

      if indices.size == 0: break

      index = indices[0]

      bits[index] -= 1
      remBits += nLines[index]    # Add lonely bit back to bit budget
      availableBands[index] = False

      indices = (SMR == max(SMR[availableBands])).nonzero()[0]

      if indices.size == 0: break

      for i in indices:
        if remBits >= nLines[i]:
          remBits -= nLines[i]
          bits[i] += 1
          if bits[i] == maxMantBits:
            availableBands[i] = False
          SMR[i] -= DBTOBITS

    return bits

    #-----------------------------------------------------------------------------

if __name__ == "__main__":
  pass


