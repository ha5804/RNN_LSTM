from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from sklearn.preprocessing import LabelEncoder
import numpy as np

class DataProcess:
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
        labels = self.log_label()
        for i in range(len(self.event_ids_encoded) - self.seq_len):
            x_seq = self.event_ids_encoded[i:i+self.seq_len]
            y_label = self.event_ids_encoded[i+self.seq_len]
            x_labels = labels[i:i+self.seq_len]     

            X.append(x_seq)
            y.append(y_label)

        X = np.array(X)  
        y = np.array(y) 
        return X , y
    
    def make_test_seq_list_2(self):
        id = self.get_event_ids()
        X = []
        y = []
        labels = self.log_label()
        for i in range(len(id) - self.seq_len):
            x_seq = id[i:i+self.seq_len]
            y_label = id[i+self.seq_len]
            x_labels = labels[i:i+self.seq_len]     

            X.append(x_seq)
            y.append(y_label)

        X = np.array(X)  
        y = np.array(y) 
        return X , y
    
    #debug==============================================
    def count_warning_sequences(self, labels, seq_len = 10):
        count = 0
        waring_index = []
        for i in range(len(labels) - seq_len):
            window = labels[i: i+seq_len]
            if any(l == 1 for l in window):
                count += 1
                waring_index.append(i)
        return count, waring_index

#===================================================
class Embedding:
    def __init__(self, event_size, embedding_dim):
        self.event_size = event_size 
        self.embedding_dim = embedding_dim
        self.embedding_matrix = np.random.randn(event_size, embedding_dim) * 0.01
    
    def forward(self, input_seq):
        return self.embedding_matrix[input_seq] 