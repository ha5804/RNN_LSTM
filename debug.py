import numpy as np
import matplotlib.pyplot as plt

#===================================================
class Embedding:
    def __init__(self, event_size, embedding_dim):
        self.event_size = event_size
        self.embedding_dim = embedding_dim
        self.embedding_matrix = np.random.randn(event_size, embedding_dim) * 0.01
    
    def forward(self, input_seq):
        return self.embedding_matrix[input_seq]

#===================================================
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
    
    def forward(self, h_prev, x_t , c_prev):
        self.concat_vector(h_prev, x_t)
        c_t = self.cell_state_update(c_prev)
        h_t = self.hidden_state(c_t)
        return h_t, c_t

 #===================================================   
class MyModel2:
    def __init__(self, feature_matrix, hidden_layer):
        self.feature_matrix = feature_matrix 
        self.weight = np.zeros((hidden_layer, 1))  

    def initialize(self, weight=None):
        if weight is not None:
            self.weight = weight
        else:
            self.weight = np.zeros_like(self.weight)

    def activate(self, x):
        return 1 / (1 + np.exp(-x))  

    def predict(self):
        z = self.feature_matrix @ self.weight  
        return self.activate(z)  

    def update(self, weight):
        self.weight = weight

    def compute_loss(self, y_true):
        y_pred = self.predict()
        m = y_true.shape[0]
        loss = -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))
        return loss

    def train(self, y_true, lr=0.01, epochs=100):
        m = y_true.shape[0]
        y_true = y_true.reshape(-1, 1)

        for epoch in range(epochs):
            y_pred = self.predict()
            grad = (self.feature_matrix.T @ (y_pred - y_true)) / m
            self.weight -= lr * grad

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {self.compute_loss(y_true):.4f}")


from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from sklearn.preprocessing import LabelEncoder
import numpy as np

class log_temp:
    def __init__(self, path = "./data/HDFS_2k.log" ,seq_len = 10):
        self.config = TemplateMinerConfig()
        self.template_miner = TemplateMiner(config = self.config)
        self.seq_len = seq_len
        self.path = path
        self.event_ids_encoded = None
        pass
        
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
    
    def make_seq_list(self):
        X = []
        y = []

        for i in range(len(self.event_ids_encoded) - self.seq_len):
            X.append(self.event_ids_encoded[i:i+self.seq_len])   
            y.append(self.event_ids_encoded[i+self.seq_len])     

        X = np.array(X)  
        y = np.array(y) 
        return X , y


model = MyModel()
data = log_temp()
data.encode_event_ids()
X_seq , y = data.make_seq_list()

event_size = len(set(data.event_ids_encoded))
embedding_dim = 16

embedding = Embedding(event_size, embedding_dim)

X_embed = np.array([embedding.forward(seq) for seq in X_seq])
print(X_embed.shape) #1990, 10 ,16

input_size = 16
hidden_size = 32
concat_size = input_size + hidden_size

w_f = np.random.randn(hidden_size, concat_size)
b_f = np.zeros((hidden_size, 1))
w_i = np.random.randn(hidden_size, concat_size)
b_i = np.zeros((hidden_size, 1))
w_c = np.random.randn(hidden_size, concat_size)
b_c = np.zeros((hidden_size, 1))
w_o = np.random.randn(hidden_size, concat_size)
b_o = np.zeros((hidden_size, 1))

model = MyModel(w_f, b_f, w_i, b_i, w_c, b_c, w_o, b_o)

h_list = []

for seq in X_embed:
    h_prev = np.zeros((hidden_size, 1))
    c_prev = np.zeros((hidden_size, 1))

    for t in range(seq.shape[0]):
        x_t = seq[t].reshape(-1, 1)
        model.get_concat_vector(h_prev, x_t)
        h_t, c_t = model.forward(h_prev, x_t, c_prev)
        h_prev, c_prev = h_t, c_t
    
    h_list.append(h_t.flatten())

h_array = np.array(h_list)


# 1. 이상 라벨 임의 생성 (예시)
y_binary = np.zeros_like(y)
y_binary[::50] = 1  # 임의로 50번째마다 이상 설정

# 2. 모델 생성 및 학습
model2 = MyModel2(h_array, hidden_size)
model2.train(y_binary, lr=0.1, epochs=100)

# 3. 예측
preds = model2.predict()
print(preds[:10].flatten())  # 처음 10개 출력
