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
        with open(self.path, "r") as f: #로그 한줄씩 읽고 특정 키워드면 1 or 0
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
    #드레인3로 로그 패턴화, 같은 종류의 로그끼리 묶음

    def encode_event_ids(self):
        encoder = LabelEncoder()
        self.event_ids_encoded = encoder.fit_transform(self.get_event_ids())
        return self.event_ids_encoded
    #clust id는 문자열이므로 정수로 변환

    def make_train_seq_list(self):
        x = []
        y = []
        labels = self.log_label()
        for i in range(len(self.event_ids_encoded) - self.seq_len):
            x_seq = self.event_ids_encoded[i:i+self.seq_len]
            y_label = self.event_ids_encoded[i+self.seq_len]
            x_labels = labels[i:i+self.seq_len]     

            if all(l == 0 for l in x_labels):
                x.append(x_seq)
                y.append(y_label)

        x = np.array(x)  
        y = np.array(y) 
        return x , y
    #학습용 트레이닝 데이터 만들기. autoencoder에서는 정상 데이터로만 학습

    def make_test_seq_list(self):
        x = []
        y = []
        labels = self.log_label()
        for i in range(len(self.event_ids_encoded) - self.seq_len):
            x_seq = self.event_ids_encoded[i:i+self.seq_len]
            y_label = self.event_ids_encoded[i+self.seq_len]
            x_labels = labels[i:i+self.seq_len]     

            x.append(x_seq)
            y.append(y_label)

        x = np.array(x)  
        y = np.array(y) 
        return x , y
    #테스트셋은 정상 이상 모두 포함.
    
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
#데이터를 대입하기위해 데이터를 저차원 벡터로 변환 event 갯수가 행으로, 차원은 우리가 설정.
#event - size = 3, embedding_dim = 5로 설정시, 3,5의 행렬 생성