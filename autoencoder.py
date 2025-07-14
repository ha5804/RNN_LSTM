import numpy as np

class Autoencoder:
    def __init__(self, input_size, hidden_size):
        #input_size == col of feature_matrix == dim of lstm hidden_size
        #hidden_size == user
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_enc = np.random.randn(hidden_size, input_size) * 0.01
        self.b_enc = np.zeros((hidden_size, 1))

        self.W_dec = np.random.randn(input_size, hidden_size) * 0.01
        self.b_dec = np.zeros((input_size, 1))
        self.train_mse_list = []
        self.threshold = None

    def encode(self, x):
        z = self.W_enc @ x + self.b_enc
        return np.tanh(z)
        #W_enc.shape = (32, 32) x.shape = (32, 1) z.shape(32,1)
    
    def decode(self, z):
        x_hat = self.W_dec @ z + self.b_dec
        return x_hat
    
    def cal_loss(self, x, x_hat):
        return np.mean((x- x_hat) ** 2)
    
    def train_data(self, feature_matrix, lr = 0.01, epochs = 100):
        for i in range(epochs):
            for row in feature_matrix:
                row = row.reshape(-1, 1)

                z = self.encode(row)
                x_hat = self.decode(z)

                d_loss = 2 * (x_hat - row) / self.input_size  

                dW_dec = d_loss @ z.T
                db_dec = d_loss

                dz = self.W_dec.T @ d_loss * (1 - z ** 2)  
                dW_enc = dz @ row.T
                db_enc = dz

                self.W_dec -= lr * dW_dec
                self.b_dec -= lr * db_dec
                self.W_enc -= lr * dW_enc
                self.b_enc -= lr * db_enc

        for row in feature_matrix:
            row = row.reshape(-1, 1)
            z = self.encode(row)
            x_hat = self.decode(z)
            loss = self.cal_loss(row, x_hat)
            self.train_mse_list.append(loss)
        
        self.threshold = np.mean(self.train_mse_list) + 3 * np.std(self.train_mse_list)

    def predict(self, new_x):
        z = self.encode(new_x)
        x_hat = self.decode(z)
        new_loss = self.cal_loss(new_x , x_hat)

        if new_loss > self.threshold:
            return "ubnomal"
        else:
            return "nomal"
        

