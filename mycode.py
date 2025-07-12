import numpy as np
import matplotlib.pyplot as plt


class Embedding:
    def __init__(self, event_size, embedding_dim):
        self.event_size = event_size
        self.embedding_dim = embedding_dim
        self.embedding_matrix = np.random.randn(event_size, embedding_dim) * 0.01
        pass


class MyModel:
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
    
    def forward(self, c_prev):
        self.concat_vector
        c_t = self.cell_state_update(c_prev)
        h_t = self.hidden_state(c_t)
        return h_t, c_t

    
