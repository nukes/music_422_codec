import numpy as np

# Question 1.b)
def BitAllocUniform(bitBudget, maxMantBits, nBands, nLines, SMR):
    """
    Return a hard-coded vector that, in the case of the signal use in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are uniformely distributed for the mantissas.
    """

    bitAllocs = np.zeros(nBands)
    mantBits = bitBudget / sum(nLines)
    if mantBits == 1:
      mantBits = 0

    if mantBits > maxMantBits:
      mantBits = maxMantBits

    bitAllocs[:] = np.round(mantBits)

    return bitAllocs

def BitAllocConstSNR(bitBudget, maxMantBits, nBands, nLines, SMR):
    """
    Return a hard-coded vector that, in the case of the signal use in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are distributed for the mantissas to try and keep a constant 
    quantization noise floor (assuming a noise floor 6 dB per bit below 
    the peak SPL line in the scale factor band).
    """

    # note: in this case, signal SPL is passed as SMR

    SPL = SMR
    Kp = len(SPL[SPL > 0])
    Rk = np.zeros(sum(nLines), dtype=np.int32)
    Rk[SPL <= 0] = 0
    trueMax = bitBudget / sum(nLines)

    mantBits = bitBudget*1.0/Kp
    if mantBits > trueMax:
      mantBits = trueMax

    Rk = np.round(mantBits + 0.5 * np.log2(SPL**2) - 1./Kp * np.sum(0.5*(np.log2((SPL[SPL>0])**2))))
    Rk[Rk<=1] = 0

    bitAllocs = np.zeros(nBands)
    n = 0
    for i in range(0,nBands):
      bitAllocs[i] = sum(Rk[n:nLines[i]+n]) 
      n += nLines[i]

    bitAllocs[bitAllocs > maxMantBits] = maxMantBits
    return bitAllocs

def BitAllocConstMNR(bitBudget, maxMantBits, nBands, nLines, SMR):
    """
    Return a hard-coded vector that, in the case of the signal use in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are distributed for the mantissas to try and keep the quantization
    noise floor a constant distance below (or above, if bit starved) the
    masked threshold curve (assuming a quantization noise floor 6 dB per
    bit below the peak SPL line in the scale factor band).
    """
    Kp = len(SMR[SMR > 0])
    Rk = np.zeros(sum(nLines), dtype=np.int32)
    Rk[SMR <= 1] = 0
    trueMax = bitBudget / sum(nLines)

    mantBits = bitBudget*1.0/Kp
    if mantBits > trueMax:
      mantBits = trueMax

    Rk = np.round(mantBits + 0.5 * np.log2(SMR**2) - 1./Kp * np.sum(0.5*(np.log2((SMR[SMR>0])**2))))
    Rk[Rk<0] = 0
    Rk[Rk>maxMantBits] = maxMantBits

    return Rk

# Question 1.c)
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
    




#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":
    
  ### Exercise 1.b ###

  # At 128k bitrate, uniform bit allocation was insufficient: clipping is audible and harsh. 
  # High frequencies are lost for constant SNR, whereas low frequencies are lost for constant
  # MNR. 
  # At 256k bitrate, uniform allocation is much closer to the original. In fact, very little 
  # difference can be perceived. 
  # Constant SNR is similar to the 128k bitrate case, whereas MNR is improved but still lacking
  # some mid-frequencies. 
  
  ### End of Exercise 1.b ###

  pass