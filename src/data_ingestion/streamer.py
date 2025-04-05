import pandas as pd
import time

def batch_generator(data, batch_size, delay):
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i: i + batch_size, :]
        yield batch
        time.sleep(delay)

# В чем разница между loc и iloc
# Как работает yield, есть ли альтернатива?
        