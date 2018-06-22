import math
import time

def _as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since):
    now = time.time()
    s = now - since
    return '%s' % _as_minutes(s)