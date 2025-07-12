import numpy as np
import matplotlib.pyplot as plt

class MyModel:
    def __init__(self, w_f, b_f):

        self.w_f = w_f
        self.b_f = b_f
        self.concat_vector = None
        pass
    
    def get_concat_vector(self, h_prev, x_t):
        self.concat_vector = np.vstack([h_prev, x_t])
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forget_gate(self):
        value = (self.w_f @ self.concat_vector) + self.b_f
        f_t = self.sigmoid(value)
        return f_t
    
    def input_gate(self):
        self.concat_vector

