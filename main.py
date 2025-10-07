from data.mnist_data import MnistData
from layers import Layers, HiddenLayer
from neural_network import NeuralNetwork
from trainer import Trainer


layers = Layers(784, 10, [HiddenLayer(100), HiddenLayer(40)])
model = NeuralNetwork(layers)

mnist_data = MnistData(16)
train_loader, test_loader = mnist_data.get_dataloaders()

trainer = Trainer(model, train_loader, test_loader, 60, 0.003)
trainer.train()