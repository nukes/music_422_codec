'''
This is a buffer class that assists with reading in WAV files. It maintains a
constant read buffer that will be adapted with the transient detector to 
form a lookahead buffer. This will help control what size window needs to be
used at any time, and provides the exact size of data for the window.
'''

from collections import deque

import numpy as np
from scipy.io.wavfile import read

from codec.onset import WindowState, onset_in_block


class TransientBuffer(object):
    ''' This is a wrapper that holds queueus. The first queue contains all of
    the WAV data (which, you know, should be changed to a streaming model, 
    but we are dealing with short files for now). The second is a lookahead
    queue which inspects the incoming data and decides how large the next
    block should be in order to obtain the correct time resolution in the
    face of transients and attack onsets.
    '''

    def __init__(self, filename, buffersize=1152):
        ''' Give a filename to have the control buffer wrap it. '''
        self.sample_rate, self._data = read(filename)

        # Normalize the 16 bit data to a range of -1...1
        # TODO: This actually might not be legit...might need to use decode instead
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

        # Create the window controller and prime it with the buffer data
        self._detects = 0
        self.window_controller = WindowState()
        self._next_window_state()


    def next(self):
        ''' Ask the buffer for the next block from the WAV file. If it is
        empty, it will return an empty list. If the lookahead buffer and the
        file buffer are at their last samples, it will zero pad the data.
        '''

        # Get the window state for the current buffer
        win_state = self.window_controller.state

        # Based on the window state, pop the correct amount of data
        if win_state == 0 or win_state == 3:
            block_size = 512
        if win_state == 1 or win_state == 2:
            block_size = 64

        # Normally, we would figure out how many blocks to emit
        # Right now, we're just going to use a magic number
        data_pop = min(block_size, len(self._data))
        buffer_pop = min(block_size, len(self._buffer))

        # Popleft the elements out of the buffer -- i.e. do a FIFO poll
        ret = []
        for x in range(buffer_pop):
            ret.append(self._buffer.popleft())
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

        # Move the window state
        self._next_window_state()

        return ret, win_state

    def _next_window_state(self):
        if len(self._buffer) > 0:
            channels = np.array(self._buffer).T
            left = onset_in_block(np.array(channels[0]))
            right = onset_in_block(np.array(channels[1]))
            self.window_controller.step(left or right)
            if self.window_controller.state == 1:
                self._detects += 1

    def __len__(self):
        ''' Remains compatible with Python len builtin. Returns the
        length of the symmetic channels, not the total number of samples
        contained therein.
        '''
        return len(self._buffer)
