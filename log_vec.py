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

    def log_label(self):
        
        
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


