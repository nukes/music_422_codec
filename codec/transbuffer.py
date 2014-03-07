'''
This is a buffer class that assists with reading in WAV files. It maintains a
constant read buffer, 
'''

from collections import deque

import numpy as np
from scipy.io.wavfile import read


class TransientBuffer(object):
    ''' Class that handles file reading from WAV files. '''

    def __init__(self, filename, buffersize=2048):
        ''' Give a file to have the control buffer wrap it. '''
        self.sample_rate, self._data = read(filename)

        # Normalize the 16 bit data to a range of -1...1
        self._data = np.array(self._data)
        self._data = self._data / (2.**15)

        # Apply useful metadata attributes
        self.channels = self._data.shape[1]
        self._samples = self._data.shape[0]

        # Finally, turn this into a destructive queue data structure so that
        # it can be popped by the transient-detecting queue
        self._data = deque(self._data)

        # Prime the FIFO with samples
        self._buffer = deque()
        for i in range(buffersize):
            self._buffer.append(self._data.popleft())

    def __len__(self):
        ''' Remains compatible with Python len builtin. Returns the
        length of the symmetic channels, not the total number of samples
        contained therein.
        '''
        return len(self._buffer)

    def next(self):
        ''' Ask the buffer for the next block from the WAV file. If it is
        empty, it will return an empty list. '''

        # Normally, we would figure out how many blocks to emit
        # Right now, we're just going to use a magic number
        block_size = 1024
        data_pop = min(block_size, len(self._data))
        buffer_pop = min(block_size, len(self._buffer))
        
        # Popleft the elements out of the buffer -- i.e. do a FIFO poll
        ret = []
        for x in range(buffer_pop):
            ret.append(self._buffer.popleft())
        #print ret
        ret = np.array(ret).T

        # Repopulate the buffer with data
        for i in range(data_pop):
            self._buffer.append(self._data.popleft())

        # If the buffer exists but is not the length we expect, zero pad
        # This cheap-ass shortcut is a precedent in the provided file :3
        if len(ret) > 0 and len(ret[0]) < block_size:
            pad_len = block_size - len(ret[0])
            ret = [np.concatenate([ret[0], np.zeros(pad_len)]),
                   np.concatenate([ret[1], np.zeros(pad_len)])]

        return ret


if __name__ == '__main__':

    # Read in the audio file
    buf = TransientBuffer('sample.wav')

    for i in range(buf._samples / 1024):
        buf.next()

