from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

config = TemplateMinerConfig()
config.load_default()
template_miner = TemplateMiner(config = config)

event_ids = []

with open("HDFS_2k.log", "r") as f:
    for line in f:
        result = template_miner.add_log_message(line.strip())
        event_id = result["cluster_id"]
        event_ids.append(event_id)

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
event_ids_encoded = encoder.fit_transform(event_ids)