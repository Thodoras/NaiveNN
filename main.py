from layers import Layers, HiddenLayer
from neural_network import NeuralNetwork


layers = Layers(50, 3, [HiddenLayer(30), HiddenLayer(20)])
model = NeuralNetwork(layers)

print(model)