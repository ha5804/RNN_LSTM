import numpy as np

class Scaler:
    def __init__(self):
        self.mean = None
        self.std = None
    
    def get_scaled_matrix(self, feature_matrix):
        self.mean = np.mean(feature_matrix, axis = 0)
        self.std = np.std(feature_matrix, axis = 0)
        feature_matrix_scaled = (feature_matrix - self.mean) / self.std
        return feature_matrix_scaled
    
    def get_scaled_test(self, test_data):
        return (test_data - self.mean) / self.std
