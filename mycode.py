import numpy as np
import matplotlib.pyplot as plt

class MyModel:
    def __init__(self, w_f, b_f):

        self.w_f = w_f
        self.b_f = b_f
        pass

    def forget_gate(self):
        