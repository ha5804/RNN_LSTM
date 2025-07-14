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




## Autoencoder

(1)definition

- Autoencoder is a method of anomaly detection by calculating the difference in reconstruction error.

(2)use lstm

- lstm acts as a feature extractor for autoencoder.

(3)procedure

- Train the autoencoder on LSTM-encoded normal sequences.

- For each test sequence, encode it using LSTM and input it into the autoencoder. 

- Measure the reconstruction error (MSE) between the original and reconstructed input.

- Compare the error to a threshold (learned from training) to determine if the input is normal or abnormal.
















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






