# RNN_LSTM

**procedure**

(1) data process

- use drain3 for parsing log data with similarity

- use sklearn for change type of data into int
    (we need int type data as LSTM)

- each log_seq is vectorized and embedded data

(2) lstm process

- log_seq is input sequence

- lstm is filtering input_data as importance
    (lstm have own standards)



















## Dataset

- The `HDFS_2k.log` file used in this project is from the [LogHub repository](https://github.com/logpai/loghub), maintained by LogPAI.
- Licensed under the Apache License 2.0.

---

## Parsing log data with drain

- TempleteMiner - classifying similar log messages into one template ID

- cluster_id - template ID with each log group

- event_ids - sequence of log order

## **Model**

 - init - each gate






