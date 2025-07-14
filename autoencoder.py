import numpy as np

class autoencoder:
    def __init__(self, input_size, hidden_size):
        #input_size == col of feature_matrix == dim of lstm hidden_size
        #hidden_size == user
        self.W_enc = np.random.randn(hidden_size, input_size) * 0.01
        self.b_enc = np.zeros((hidden_size, 1))

        self.W_dec = np.random.randn(input_size, hidden_size) * 0.01
        self.b_dec = np.zeros((input_size, 1))

    def encode(self, x):
        z = self.W_enc @ x + self.b_enc
        return np.tanh(z)
        #W_enc.shape = (32, 32) x.shape = (32, 1) z.shape(32,1)
    
    def decode(self, z):
        x_hat = self.W_dec @ z + self.b_dec
        return x_hat
    
    def cal_loss(self, x, x_hat):
        return np.mean((x- x_hat) ** 2)
    
