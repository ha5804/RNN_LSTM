import matplotlib.pyplot as plt
import seaborn as sns
class MyPlot:
    def __init__(self):
        self.figsize = (8,8)
        pass

    def sliding_window_data(self, x_sequence):
        plt.figure(figsize = self.figsize)
        plt.xlabel("time_table")
        plt.ylabel("number of sample")
        plt.scatter(range(len(x_sequence)),x_sequence, marker = '.', color = 'r')
        plt.show()