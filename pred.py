# x = [log ID seq] -> (y = next event ID)
#  predict with probability
import numpy as np 
from mycode import LSTM_CELL
class LSTMPredict:
    def __init__(self, number_log_seq, hidden_size, embedding_dim, seq_len):
        self.number_log_seq = number_log_seq
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        
        self.embeddings = np.random.randn(number_log_seq, embedding_dim) * 0.1

        weight_shape = (hidden_size, hidden_size + embedding_dim)
        self.lstm = LSTM_CELL(w_f=np.random.randn(*weight_shape) * 0.1,
            b_f=np.zeros((hidden_size, 1)),
            w_i=np.random.randn(*weight_shape) * 0.1,
            b_i=np.zeros((hidden_size, 1)),
            w_c=np.random.randn(*weight_shape) * 0.1,
            b_c=np.zeros((hidden_size, 1)),
            w_o=np.random.randn(*weight_shape) * 0.1,
            b_o=np.zeros((hidden_size, 1))
        )

        self.W_out = np.random.randn(number_log_seq, hidden_size) * 0.1
        self.b_out = np.zeros((number_log_seq, 1))


