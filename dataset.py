import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_circles

class Data:

    def __init__(self, nData):
        self.nData = nData

    def buildDataInCircles(self):
        self.X, self.Y = make_circles(n_samples = self.nData, factor = 0.5, noise = 0.06)
        self.Y = self.Y[:, np.newaxis]


    def showDataConsole(self):
        print(f'=> X = {self.X} \n=>Y = {self.Y} ')

    def showDataGraph(self):
        plt.scatter(self.X[self.Y[:, 0] == 0, 0], self.X[self.Y[:, 0] == 0, 1], c = "skyblue")
        plt.scatter(self.X[self.Y[:, 0] == 1, 0], self.X[self.Y[:, 0] == 1, 1], c = "salmon")
        plt.axis("equal")
        plt.show()

    def getData(self):
        return self.X, self.Y



