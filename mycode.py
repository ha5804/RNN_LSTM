import numpy as np
import matplotlib.pyplot as plt

class MyModel:
    def __init__(self, w_f, b_f):

        self.w_f = w_f
        self.b_f = b_f
        pass
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forget_gate(self, h_prev, x_t):
        concat_vector = np.vstack([h_prev, x_t])
        
        f_t = self.sigmoid()