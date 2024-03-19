import numpy as np

import tensorflow.keras as keras

class KerasSequence(keras.utils.Sequence):
    def __init__(self, entries:list, batch_size:int, x_function, y_function):
        self.entries = entries
        self.batch_size = batch_size
        self.x_function = x_function
        self.y_function = y_function
    
    def __len__(self):
        return (np.ceil(len(self.entries) / float(self.batch_size))).astype(np.int64)
    
    def __getitem__(self, idx):
        batch_x = [self.x_function(entry) for entry in self.entries[idx * self.batch_size : (idx+1) * self.batch_size]]
        batch_y = [self.y_function(entry) for entry in self.entries[idx * self.batch_size : (idx+1) * self.batch_size]]
        
        return np.array(batch_x), np.array(batch_y)
