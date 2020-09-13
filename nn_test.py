import dataset
import neural_network

import time

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt


from sklearn.datasets import make_circles
from IPython.display import clear_output

nData = 500
nFeaturesPattern = 2

dt = dataset.Data(nData)
dt.buildDataInCircles()
# dt.showDataGraph()
X, Y = dt.getData()

topology = [nFeaturesPattern, 4, 8, 1]
nn = neural_network.NN(topology)

iteration = 25000
loss = [1]

for i in range(iteration):
  pY = nn.trainingNeuralNetwork(X, Y, learning_factor = 0.5)
  error = nn.functionCost[0](nn.out[-1][1], Y)

  if error < loss[-1]:
    print(f'itr = {i} | error = {error} | status = <')
  else:
    print(f'itr = {i} | error = {error} | status = >')

  loss.append(error)
  time.sleep(0.1)
