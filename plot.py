import matplotlib.pyplot as plt
import seaborn as sns
class MyPlot:
    def __init__(self):
        self.figsize = (5,5)
        pass

    def sliding_window_data(self, x_sequence, label):
        plt.figure(figsize = self.figsize)
        plt.xlabel("number of log_sample")
        plt.ylabel("int value of log_sample")
        plt.scatter(range(len(x_sequence)),x_sequence, s = 0.1, color = 'r')
        plt.scatter(range(len(label)), label, s = 0.1, color = 'b')
        plt.show()
    
