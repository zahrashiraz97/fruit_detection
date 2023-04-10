from collections import deque
import numpy as np
import time


class Stopwatch:
    def __init__(self, buffer_size=32):
        self.prev_time = None
        self.times = deque(maxlen=buffer_size)

    def start(self):
        self.prev_time = time.time()

    def lap(self):
        curr_time = time.time()
        if self.prev_time is not None:
            self.times.append(curr_time - self.prev_time)
        self.prev_time = curr_time

    def last_time(self):
        return self.times[-1] if len(self.times) > 0 else None

    def average_time(self):
        return np.mean(self.times)
