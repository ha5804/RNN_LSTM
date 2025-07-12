from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

config = TemplateMinerConfig()

template_miner = TemplateMiner(config = config)

event_ids = []

with open("./data/HDFS_2k.log", "r") as f:
    for line in f:
        result = template_miner.add_log_message(line.strip())
        event_id = result["cluster_id"]
        event_ids.append(event_id)

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
event_ids_encoded = encoder.fit_transform(event_ids)

import numpy as np

seq_len = 10 
X = []
y = []

for i in range(len(event_ids_encoded) - seq_len):
    X.append(event_ids_encoded[i:i+seq_len])   
    y.append(event_ids_encoded[i+seq_len])     

X = np.array(X)  
y = np.array(y) 
