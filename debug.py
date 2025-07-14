from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt


class log_temp:
    def __init__(self, path = "./data/HDFS_2k.log" ,seq_len = 10):
        self.config = TemplateMinerConfig()
        self.template_miner = TemplateMiner(config = self.config)
        self.seq_len = seq_len
        self.path = path
        self.event_ids_encoded = self.encode_event_ids()
        pass

    def log_label(self):
        label = []
        with open(self.path, "r") as f:
            for line in f:
                if any(keyword in line for keyword in ['WARN', 'ERROR', "FATAL"]):
                    label.append(1)
                else:
                    label.append(0)
        return label
            
    def get_event_ids(self):
        event_ids = []

        with open(self.path, "r") as f:
            for line in f:
                result = self.template_miner.add_log_message(line.strip())
                event_id = result["cluster_id"]
                event_ids.append(event_id)
        return event_ids
    
    def encode_event_ids(self):
        encoder = LabelEncoder()
        self.event_ids_encoded = encoder.fit_transform(self.get_event_ids())
        return self.event_ids_encoded
    
    def make_train_seq_list(self):
        X = []
        y = []
        labels = self.log_label()
        for i in range(len(self.event_ids_encoded) - self.seq_len):
            x_seq = self.event_ids_encoded[i:i+self.seq_len]
            y_label = self.event_ids_encoded[i+self.seq_len]
            x_labels = labels[i:i+self.seq_len]     

            if all(l == 0 for l in x_labels):
                X.append(x_seq)
                y.append(y_label)

        X = np.array(X)  
        y = np.array(y) 
        return X , y
    
    def make_test_seq_list(self):
        X = []
        y = []
        
        for i in range(len(self.event_ids_encoded) - self.seq_len):
            x_seq = self.event_ids_encoded[i:i+self.seq_len]
            y_label = self.event_ids_encoded[i+self.seq_len]   

            X.append(x_seq)
            y.append(y_label)

        X = np.array(X)  
        y = np.array(y) 
        return X , y

#===================================================
class Embedding:
    def __init__(self, event_size, embedding_dim):
        self.event_size = event_size 
        self.embedding_dim = embedding_dim
        self.embedding_matrix = np.random.randn(event_size, embedding_dim) * 0.01
    
    def forward(self, input_seq):
        return self.embedding_matrix[input_seq] 
    
#===================================================
class LSTM_CELL:
    def __init__(self, w_f, b_f, w_i, b_i, w_c, b_c, w_o, b_o):

        self.w_f = w_f
        self.b_f = b_f
        self.w_i = w_i
        self.b_i = b_i
        self.w_c = w_c
        self.b_c = b_c
        self.w_o = w_o
        self.b_o = b_o
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
        value = (self.w_i @ self.concat_vector) + self.b_i
        i_t = self.sigmoid(value)
        return i_t
    
    def candidate_memory(self):
        value = (self.w_c @ self.concat_vector) + self.b_c
        c_t_hat = np.tanh(value)
        return c_t_hat
    

    def cell_state_update(self,c_prev):
        c_t_hat = self.candidate_memory()
        f_t = self.forget_gate()
        i_t = self.input_gate()
        c_t = (f_t * c_prev) + (i_t * c_t_hat)
        return c_t
    
    def output_gate(self):
        value = (self.w_o @ self.concat_vector) + self.b_o
        o_t = self.sigmoid(value)
        return o_t

    def hidden_state(self, c_t):
        h_t = self.output_gate() * np.tanh(c_t)
        return h_t
    
    def forward(self, h_prev, x_t , c_prev):
        self.get_concat_vector(h_prev, x_t)
        c_t = self.cell_state_update(c_prev)
        h_t = self.hidden_state(c_t)
        return h_t, c_t


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

    def get_threshold(self):
        return self.threshold

    def predict(self, new_x):
        z = self.encode(new_x)
        x_hat = self.decode(z)
        new_loss = self.cal_loss(new_x , x_hat)

        if new_loss > self.threshold:
            return new_loss, "abnormal"
        else:
            return new_loss, "normal"